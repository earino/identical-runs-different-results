"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: base columns + cyclic time encodings + smoothed target encodings of categorical x
time-bin interactions (fit on TRAIN ONLY, out-of-fold for training rows). Final predictor is the
average of a small ensemble of XGBoost models built on differently-shuffled OOF folds (fold-varied
target-encoding noise decorrelates the members).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering -------------------------------------------------------
BASE_CAT = ["UniqueCarrier"]
OH_COLS = ["Month", "DayofMonth", "DayOfWeek", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CAT}
oh_levels = {c: sorted(train[c].dropna().unique()) for c in OH_COLS}

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_train.mean())
SM = 35.0  # smoothing for target encoding

N_MODELS = 7
FOLD_SEEDS = [42, 1, 2, 3, 4, 5, 6]
MODEL_SEEDS = [42, 42, 7, 13, 17, 23, 31]
MEMBER_KW = [
    {"max_depth": 6, "learning_rate": 0.03, "colsample_bytree": 0.6, "subsample": 0.85},
] * N_MODELS


def _num(s):
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def _bin(df, minutes):
    return (df["DepTime"].astype(int) // minutes).astype(str)


def _b20(df):
    return _bin(df, 20)


def _b15(df):
    return _bin(df, 15)


def _distbin(df):
    return (df["Distance"].astype(int) // 300).astype(str)


# interaction keys, all derived from raw columns only
TE_SPECS = {
    "routeH": lambda d: d["Origin"].astype(str) + "_" + d["Dest"].astype(str) + "_" + _b20(d),
    "carrierH": lambda d: d["UniqueCarrier"].astype(str) + "_" + _b20(d),
    "originH": lambda d: d["Origin"].astype(str) + "_" + _b15(d),
    "destH": lambda d: d["Dest"].astype(str) + "_" + _b15(d),
    "distH": lambda d: _distbin(d) + "_" + _b20(d),
    "carrierOrg": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Origin"].astype(str),
    "carrierDest": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Dest"].astype(str),
    "routeH60": lambda d: d["Origin"].astype(str) + "_" + d["Dest"].astype(str) + "_" + _bin(d, 60),
    "routePlain": lambda d: d["Origin"].astype(str) + "_" + d["Dest"].astype(str),
    "originPlain": lambda d: d["Origin"].astype(str),
    "destPlain": lambda d: d["Dest"].astype(str),
    "carrierPlain": lambda d: d["UniqueCarrier"].astype(str),
    "distPlain": lambda d: _distbin(d),
}


def _group_stats(keys, y):
    g = pd.DataFrame({"c": pd.Series(keys).to_numpy(), "y": y}).groupby("c")["y"].agg(["sum", "count"])
    return g["sum"], g["count"]


# full-train smoothed TE + count maps (fit on train only)
TE_MAPS = {}
for name, kf in TE_SPECS.items():
    s, c = _group_stats(kf(train), y_train)
    TE_MAPS[name] = ((s + SM * PRIOR) / (c + SM), c)

# out-of-fold TEs for the training rows, one shuffle per ensemble member
OOF = {fs: {} for fs in FOLD_SEEDS}
for fs in FOLD_SEEDS:
    rng = np.random.RandomState(fs)
    folds = np.array_split(rng.permutation(len(train)), 5)
    for name, kf in TE_SPECS.items():
        keys = pd.Series(kf(train))
        oof = np.full(len(train), PRIOR, dtype=float)
        for f in folds:
            mask = np.ones(len(train), bool)
            mask[f] = False
            s, c = _group_stats(keys.to_numpy()[mask], y_train[mask])
            oof[f] = keys.iloc[f].map((s + SM * PRIOR) / (c + SM)).fillna(PRIOR).to_numpy()
        OOF[fs][name] = oof


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame (any subset of train's columns, target optional) -> feature frame."""
    X = pd.DataFrame(index=df.index)
    for c in BASE_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dt = df["DepTime"].astype(int)
    X["DepTime"] = dt.astype(float)
    X["Distance"] = df["Distance"].astype(float)
    h = dt // 100
    min_of_day = (h * 60 + dt % 100).astype(float)
    X["Hour"] = h.astype(float)
    X["MinOfDay"] = min_of_day
    X["sin_h"] = np.sin(2 * np.pi * min_of_day / 1440)
    X["cos_h"] = np.cos(2 * np.pi * min_of_day / 1440)
    doy = (_num(df["Month"]) * 31 + _num(df["DayofMonth"])).astype(float)
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    X["Bin20"] = (dt // 20).astype(float)
    oh_parts = [X]
    for c in OH_COLS:
        for v in oh_levels[c]:
            oh_parts.append(pd.DataFrame({f"OH_{c}_{v}": (df[c] == v).astype(float).to_numpy()}, index=df.index))
    te_parts = []
    for name, kf in TE_SPECS.items():
        k = pd.Series(kf(df), index=df.index)
        te_map, cnt_map = TE_MAPS[name]
        te_parts.append(pd.DataFrame({
            "TE_" + name: k.map(te_map).fillna(PRIOR).to_numpy(),
            "N_" + name: k.map(cnt_map).fillna(0).astype(float).to_numpy(),
        }, index=df.index))
    X = pd.concat(oh_parts + te_parts, axis=1)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble -----------------------------------------------------------------
def _base_frame():
    return prepare(train)


def _model_params(seed, kw):
    p = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        eval_metric="auc",
    )
    p.update(kw)
    return p


t0 = time.time()
models = []
X_base = _base_frame()
for fs, ms, kw in zip(FOLD_SEEDS, MODEL_SEEDS, MEMBER_KW):
    X_fit = X_base.copy()
    for name in TE_SPECS:
        X_fit["TE_" + name] = OOF[fs][name]
    m = xgb.XGBClassifier(**_model_params(ms, kw))
    m.fit(X_fit, y_train, verbose=False)
    models.append(m)
    del X_fit
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    return ps.mean(axis=1)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
