"""XGBoost binary classifier for flight delay prediction.

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

# categorical (native) features kept from the raw schema
CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# numeric features (DepTime/Distance raw + derived day-of-year)
NUM_COLS = ["DepTime", "Distance", "doy2", "hour", "depmin"]
# columns that get smoothed target encoding (fit on train only)
TE_COLS = ["Origin", "Dest", "UniqueCarrier", "route", "oh", "dh", "ch", "co", "mh", "da", "ca"]
# columns that get frequency encoding (fit on train only)
FREQ_COLS = ["route", "Origin", "Dest", "UniqueCarrier"]
SMOOTH = 20.0
SEEDS = (42, 7, 2024)
MONTH_CUM = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    s = lambda c: d[c].astype(str)
    h = (d["DepTime"] // 100 % 24).astype(str)
    mo = d["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    do = d["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    d["doy2"] = (MONTH_CUM[mo] + do).astype(float)
    d["hour"] = (d["DepTime"] // 100 % 24).clip(0, 23).astype(float)
    d["depmin"] = (d["DepTime"] % 100).astype(float)
    d["route"] = s("Origin") + "_" + s("Dest")
    d["oh"] = s("Origin") + "_" + h
    d["dh"] = s("Dest") + "_" + h
    arr = (d["hour"] + d["Distance"] / 450.0).round().clip(0, 29).astype(int).astype(str)
    d["da"] = s("Dest") + "_" + arr
    d["ca"] = s("UniqueCarrier") + "_" + arr
    d["ch"] = s("UniqueCarrier") + "_" + h
    d["co"] = s("UniqueCarrier") + "_" + s("Origin")
    d["mh"] = d["Month"].astype(str).str.replace("c-", "", regex=False) + "_" + h
    return d


train = add_derived(pd.read_csv("data/train.csv"))
evald = add_derived(pd.read_csv("data/eval.csv"))
train["_y"] = (train[TARGET] == POSITIVE).astype(int)
y_train = train["_y"].to_numpy()

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
PRIOR = float(y_train.mean())


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- encoders fitted on training data only ------------------------------------
def _te_map(frame: pd.DataFrame, col: str) -> pd.Series:
    g = frame.groupby(col, observed=True)["_y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)


te_maps = {c: _te_map(train, c) for c in TE_COLS}
freq_maps = {c: train[c].value_counts() for c in FREQ_COLS}

# out-of-fold target encodings for the training rows (avoid target leakage)
oof = np.full((len(train), len(TE_COLS)), PRIOR, dtype=float)
kf = KFold(n_splits=5, shuffle=True, random_state=1)
for tr_idx, va_idx in kf.split(train):
    sub = train.iloc[tr_idx]
    for j, c in enumerate(TE_COLS):
        oof[va_idx, j] = train.iloc[va_idx][c].map(_te_map(sub, c)).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    d = add_derived(df)
    X = d[CAT_COLS + NUM_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in FREQ_COLS:
        X[c + "_freq"] = d[c].map(freq_maps[c]).astype(float).fillna(0.0)
    for c in TE_COLS:
        X[c + "_te"] = d[c].map(te_maps[c]).astype(float).fillna(PRIOR)
    return X


X_train = prepare(train)
for j, c in enumerate(TE_COLS):
    X_train[c + "_te"] = oof[:, j]

# --- model: small ensemble of regularized XGBoost trees -----------------------
models = []
t0 = time.time()
for s in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        reg_lambda=1.0,
        reg_alpha=1.0,
        gamma=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
