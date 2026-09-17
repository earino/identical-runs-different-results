"""XGBoost binary classifier for airline delays (agent-edited file).

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
cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime"] = dt
    hour = (dt // 100) % 24
    minute = dt % 100
    tod = (hour * 60 + minute) / 1440.0
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    X["early_morning"] = ((hour >= 5) & (hour < 9)).astype(float)
    X["evening"] = (hour >= 17).astype(float)
    X["redeye"] = ((hour < 6) | (hour >= 21)).astype(float)
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    X["dayofmonth"] = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["dayofweek"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
configs = [
    dict(n_estimators=1000, max_depth=5, learning_rate=0.015, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(n_estimators=1000, max_depth=6, learning_rate=0.015, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(n_estimators=1000, max_depth=6, learning_rate=0.015, subsample=0.9, colsample_bytree=0.6, min_child_weight=10),
    dict(n_estimators=800, max_depth=7, learning_rate=0.02, subsample=0.7, colsample_bytree=0.9, min_child_weight=5),
    dict(n_estimators=1200, max_depth=5, learning_rate=0.012, subsample=0.7, colsample_bytree=0.7, min_child_weight=8),
    dict(n_estimators=800, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, min_child_weight=3),
    dict(n_estimators=2000, max_depth=6, learning_rate=0.008, subsample=0.75, colsample_bytree=0.75, min_child_weight=5),
    dict(n_estimators=1500, max_depth=5, learning_rate=0.01, subsample=0.85, colsample_bytree=0.85, min_child_weight=5),
]
models = []
for i, cfg in enumerate(configs):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
                          random_state=42 + i, **cfg)
    m.fit(X_train, y_train, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
