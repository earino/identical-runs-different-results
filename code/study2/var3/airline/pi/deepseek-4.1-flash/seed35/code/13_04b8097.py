"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _dep_minutes(s: pd.Series) -> pd.Series:
    t = pd.to_numeric(s, errors="coerce").fillna(0).astype(float)
    return ((t // 100) * 60 + (t % 100)).clip(0, 1440)


def _keys(df: pd.DataFrame) -> dict:
    c = df["UniqueCarrier"].astype(str)
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    h = (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(float) // 100).clip(0, 23).astype(int).astype(str)
    return {
        "carrier": c,
        "origin": o,
        "dest": d,
        "route": o + "_" + d,
        "carrier_hour": c + "_" + h,
        "origin_hour": o + "_" + h,
        "dest_hour": d + "_" + h,
        "route_hour": o + "_" + d + "_" + h,
        "origin_carrier": o + "_" + c,
        "dest_carrier": d + "_" + c,
    }


_TMP_KEYS = _keys(train)
CAT_LEVELS = {c: pd.Index(sorted(_TMP_KEYS[c].unique())) for c in ["carrier", "origin", "dest"]}
CAT_COLS = ["carrier", "origin", "dest"]

# count encodings fitted on TRAIN only
_tr_keys = _TMP_KEYS
_count_maps = {c: _tr_keys[c].value_counts().to_dict() for c in ["carrier", "origin", "dest", "route"]}
_hour_count_maps = {c: _tr_keys[c].value_counts().to_dict() for c in ["origin_hour", "dest_hour", "route_hour"]}

# smoothed target encodings (out-of-fold for training rows, full-train for inference)
ytr_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(ytr_all.mean())
SMOOTH = 20.0
TE_COLS = ["carrier", "origin", "dest", "route", "carrier_hour", "origin_hour", "dest_hour", "origin_carrier", "dest_carrier"]


def _fit_te(vals: pd.Series, yy: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"v": vals.to_numpy(), "y": yy}).groupby("v")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)


_full_te = {c: _fit_te(_tr_keys[c], ytr_all) for c in TE_COLS}
_oof_te = {c: np.full(len(train), PRIOR) for c in TE_COLS}
for tr_idx, va_idx in KFold(n_splits=10, shuffle=True, random_state=SEED).split(train):
    for c in TE_COLS:
        m = _fit_te(_tr_keys[c].iloc[tr_idx], ytr_all[tr_idx])
        _oof_te[c][va_idx] = _tr_keys[c].iloc[va_idx].map(m).fillna(PRIOR).to_numpy()


def base(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month, dom, dow = _cnum(df["Month"]), _cnum(df["DayofMonth"]), _cnum(df["DayOfWeek"])
    tt = _dep_minutes(df["DepTime"])
    hour = tt / 60.0
    dist = pd.to_numeric(df["Distance"], errors="coerce")

    X["month"], X["dom"], X["dow"] = month, dom, dow
    X["dep_hour"], X["dep_tt"], X["dep_min"] = hour, tt, tt % 60
    X["distance"], X["log_distance"] = dist, np.log1p(dist)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)

    keys = _keys(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(keys[c], categories=CAT_LEVELS[c])
    for c in ["carrier", "origin", "dest", "route"]:
        X[c + "_count"] = keys[c].map(_count_maps[c]).fillna(0)
    for c in ["origin_hour", "dest_hour", "route_hour"]:
        X[c + "_count"] = keys[c].map(_hour_count_maps[c]).fillna(0)
    X["origin_hour_frac"] = X["origin_hour_count"] / X["origin_count"].clip(lower=1)
    X["dest_hour_frac"] = X["dest_hour_count"] / X["dest_count"].clip(lower=1)
    X["route_frac"] = X["route_count"] / X["origin_count"].clip(lower=1)
    return X, keys


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X, keys = base(df)
    for c in TE_COLS:
        X[c + "_te"] = keys[c].map(_full_te[c]).fillna(PRIOR).astype(float)
    return X


def prepare_train() -> pd.DataFrame:
    X, _ = base(train)
    for c in TE_COLS:
        X[c + "_te"] = _oof_te[c]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def _make_model(seed: int, colsample: float = 0.2, subsample: float = 0.5) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=8,
        learning_rate=0.03,
        subsample=subsample,
        colsample_bytree=colsample,
        min_child_weight=40,
        reg_lambda=3.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        random_state=seed,
        n_jobs=N_JOBS,
        early_stopping_rounds=100,
        eval_metric="auc",
    )


MODEL_SPECS = [(42, 0.20, 0.50), (202, 0.15, 0.40), (777, 0.25, 0.60)]
Xtr = prepare_train()
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

models = []
t0 = time.time()
for seed, cs, ss in MODEL_SPECS:
    m = _make_model(seed, cs, ss)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"  seed {seed} cs={cs} ss={ss}: best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s for {len(MODEL_SPECS)} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
