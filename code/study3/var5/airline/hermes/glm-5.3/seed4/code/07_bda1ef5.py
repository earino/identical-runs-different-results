"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# interaction categorical levels (train only)
_hh = pd.Series(np.floor(np.asarray(pd.to_numeric(train["DepTime"], errors="coerce"), dtype="float64") / 100.0) % 24, index=train.index)
carrier_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_h" + _hh.astype("Int64").astype(str)).dropna().unique()))

# raw column names usable at predict time
RAW = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
RAW = [c for c in RAW if c in train.columns and c != TARGET]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # --- engineered numeric features from raw columns ---
    def g(col):
        if col in df.columns:
            return np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype="float64")
        return np.full(len(df), np.nan)

    mon = g("Month")
    dom = g("DayofMonth")
    dow = g("DayOfWeek")
    dep = g("DepTime")
    dist = g("Distance")

    X["f_dep_time"] = dep  # raw hhmm integer
    X["f_dep_hour"] = np.floor(dep / 100) % 24
    X["f_dep_minute"] = dep % 100
    X["f_dep_frac"] = (dep % 2400) / 24.0  # hhmm as fraction of day (careful: 2400 wraps)
    X["f_dep_norm"] = dep / 2400.0
    X["f_dep_sind"] = np.sin(2 * np.pi * dep / 2400.0)
    X["f_dep_cosd"] = np.cos(2 * np.pi * dep / 2400.0)
    X["f_dom"] = dom
    X["f_dow"] = dow
    X["f_month"] = mon
    X["f_dist"] = dist
    X["f_dist_log"] = np.log1p(dist)
    # minute-of-day: a cleaner continuous clock than hhmm
    hh = np.floor(dep / 100.0)
    X["f_minOfDay"] = hh * 60.0 + (dep - 100.0 * hh)
    # interaction categorical: carrier x hour-of-day
    if "UniqueCarrier" in df.columns:
        hhs = pd.Series(hh % 24, index=df.index).astype("Int64").astype(str)
        hc = df["UniqueCarrier"].astype(str) + "_h" + hhs
        X["f_carrier_hour"] = pd.Categorical(hc, categories=carrier_hour_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed ensemble (XGBoost only, allowed) ------------------------------
N_MODELS = 5
t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
models = []
CFG = [
    dict(max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0),
    dict(max_depth=7, subsample=0.7, colsample_bytree=0.7, min_child_weight=10, reg_lambda=3.0),
    dict(max_depth=9, subsample=0.9, colsample_bytree=0.6, min_child_weight=3, reg_lambda=1.0),
    dict(max_depth=6, subsample=0.75, colsample_bytree=0.9, min_child_weight=8, reg_lambda=2.0),
    dict(max_depth=10, subsample=0.85, colsample_bytree=0.75, min_child_weight=6, reg_lambda=4.0),
]
for k in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + k,
        n_jobs=N_JOBS,
        **CFG[k],
    )
    m.fit(Xtr, ytr)
    models.append(m)
model = models[0]
print(f"Training time: {time.time() - t0:.1f}s for {N_MODELS} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
