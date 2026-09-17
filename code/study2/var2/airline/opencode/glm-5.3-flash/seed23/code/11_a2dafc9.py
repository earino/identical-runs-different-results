"""XGBoost ensemble (multi-view) for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering -------------------------------------------------------

FEAT_NUM = ["DepHour", "DepMinute", "Distance"]
FEAT_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "CarrierHour"]
TE_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "DayOfWeek", "HourCat", "Month"]
SMOOTH = 100


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering only (no fitted statistics)."""
    X = df.copy()
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["DepHour"] = (dep // 100) % 24
    X["DepMinute"] = dep % 100
    X["HourCat"] = X["DepHour"].astype(str)
    X["CarrierHour"] = X["UniqueCarrier"].astype(str) + "_" + X["DepHour"].astype(str)
    X["CarrierSlot30"] = X["UniqueCarrier"].astype(str) + "_" + (dep // 30).astype(str)
    X["CarrierSlot15"] = X["UniqueCarrier"].astype(str) + "_" + (dep // 15).astype(str)
    X["OrigHour"] = X["Origin"].astype(str) + "_" + X["DepHour"].astype(str)
    X["Route"] = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    return X


_tr_fe = add_base_features(train)
levels = {c: pd.Index(sorted(_tr_fe[c].dropna().unique())) for c in _tr_fe.columns if _tr_fe[c].dtype == object}

# target-encoding maps, fit on train only
ytr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(ytr.mean())


def _te_maps(df_fe, y, smooth=SMOOTH):
    out = {}
    for c in TE_COLS:
        g = pd.DataFrame({"c": df_fe[c].values, "y": y}).groupby("c")["y"].agg(["mean", "count"])
        out[c] = (g["mean"] * g["count"] + prior * smooth) / (g["count"] + smooth)
    return out


def _te_apply(df_fe, maps):
    return {c: df_fe[c].map(maps[c]).fillna(prior).astype(float).to_numpy() for c in TE_COLS}


te_full = _te_maps(_tr_fe, ytr)
te_oof = np.zeros((len(_tr_fe), len(TE_COLS)))
_kf = KFold(5, shuffle=True, random_state=SEED)
for _tr_i, _va_i in _kf.split(np.arange(len(_tr_fe))):
    _enc = _te_apply(_tr_fe.iloc[_va_i], _te_maps(_tr_fe.iloc[_tr_i], ytr[_tr_i]))
    te_oof[_va_i] = np.stack([_enc[c] for c in TE_COLS], axis=1)


def prepare(df: pd.DataFrame, view: str = "main") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    F = add_base_features(df)
    X = F[FEAT_NUM + FEAT_CAT].copy()
    for c in FEAT_CAT:
        X[c] = pd.Categorical(X[c], categories=levels[c])  # unseen levels -> NaN
    if view == "hourcat":
        X = X.drop(columns=["CarrierHour"])
        X["HourCat"] = pd.Categorical(F["HourCat"], categories=levels["HourCat"])
    elif view == "te":
        enc = _te_apply(F, te_full)
        for c in TE_COLS:
            X["te_" + c] = enc[c]
    elif view == "route":
        X["Route"] = pd.Categorical(F["Route"], categories=levels["Route"])
    elif view == "cslot30":
        X["CarrierSlot30"] = pd.Categorical(F["CarrierSlot30"], categories=levels["CarrierSlot30"])
    elif view == "cslot15":
        X["CarrierSlot15"] = pd.Categorical(F["CarrierSlot15"], categories=levels["CarrierSlot15"])
    elif view in ("slotmain", "slotmain30"):
        col = "CarrierSlot15" if view == "slotmain" else "CarrierSlot30"
        X["CarrierHour"] = pd.Categorical(F[col], categories=levels[col])
    elif view == "orighour":
        X["OrigHour"] = pd.Categorical(F["OrigHour"], categories=levels["OrigHour"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of diverse configs across feature views --------------------
CONFIGS = [
    ("main", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("main", dict(n_estimators=800, max_depth=5, learning_rate=0.04, colsample_bylevel=0.3)),
    ("main", dict(n_estimators=800, max_depth=5, learning_rate=0.04, reg_alpha=1.0)),
    ("main", dict(n_estimators=600, max_depth=6, learning_rate=0.05, colsample_bylevel=0.5)),
    ("main", dict(n_estimators=800, max_depth=5, learning_rate=0.04, colsample_bylevel=0.5, subsample=0.9)),
    ("main", dict(n_estimators=300, max_depth=9, learning_rate=0.06, colsample_bylevel=0.3)),
    ("hourcat", dict(n_estimators=800, max_depth=5, learning_rate=0.04, colsample_bylevel=0.3)),
    ("hourcat", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("te", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("te", dict(n_estimators=600, max_depth=6, learning_rate=0.05, colsample_bylevel=0.5)),
    ("route", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("cslot30", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("cslot15", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("cslot15", dict(n_estimators=600, max_depth=6, learning_rate=0.05, colsample_bylevel=0.5)),
    ("slotmain30", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("slotmain", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
    ("orighour", dict(n_estimators=500, max_depth=7, learning_rate=0.05, colsample_bylevel=0.5)),
]

t0 = time.time()
Xtr_views = {}
models = []
for i, (view, cfg) in enumerate(CONFIGS):
    if view not in Xtr_views:
        Xv = prepare(train, view)
        if view == "te":
            for j, c in enumerate(TE_COLS):
                Xv["te_" + c] = te_oof[:, j]  # OOF encodings for training rows
        Xtr_views[view] = Xv
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **cfg)
    m.fit(Xtr_views[view], ytr)
    models.append((view, m))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    cache = {}
    preds = []
    for view, m in models:
        if view not in cache:
            cache[view] = prepare(df, view)
        preds.append(m.predict_proba(cache[view])[:, 1])
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
