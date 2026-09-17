"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = X["DepTime"].to_numpy(dtype=float)
    dep = np.where(dep > 2400, np.nan, dep)
    hour = np.floor(dep / 100.0)
    minute = dep % 100
    minute = np.where(minute > 59, np.nan, minute)
    tod = hour + minute / 60.0
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    dist = X["Distance"].to_numpy(dtype=float)
    arr_tod = np.mod(tod + dist / 450.0 + 0.5, 24.0)
    X["arr_hour"] = np.floor(arr_tod)
    X["arr_sin"] = np.sin(2 * np.pi * arr_tod / 24.0)
    X["arr_cos"] = np.cos(2 * np.pi * arr_tod / 24.0)
    X["tod_sin2"] = np.sin(4 * np.pi * tod / 24.0)
    X["tod_cos2"] = np.cos(4 * np.pi * tod / 24.0)
    X["arr_sin2"] = np.sin(4 * np.pi * arr_tod / 24.0)
    X["arr_cos2"] = np.cos(4 * np.pi * arr_tod / 24.0)
    X["tod_sin3"] = np.sin(6 * np.pi * tod / 24.0)
    X["tod_cos3"] = np.cos(6 * np.pi * tod / 24.0)
    X["arr_sin3"] = np.sin(6 * np.pi * arr_tod / 24.0)
    X["arr_cos3"] = np.cos(6 * np.pi * arr_tod / 24.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=3, n_estimators=1200, learning_rate=0.05),
    dict(max_depth=4, n_estimators=800, learning_rate=0.05),
    dict(max_depth=5, n_estimators=900, learning_rate=0.03),
    dict(max_depth=6, n_estimators=600, learning_rate=0.04),
    dict(max_depth=7, n_estimators=500, learning_rate=0.03),
    dict(max_depth=8, n_estimators=400, learning_rate=0.03),
    dict(max_depth=10, n_estimators=300, learning_rate=0.03),
    dict(max_depth=12, n_estimators=250, learning_rate=0.03),
    dict(max_depth=6, n_estimators=600, learning_rate=0.04, subsample=0.7,
         colsample_bytree=0.7, random_state=7),
]
models = [
    xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        monotone_constraints={"dep_hour": 1, "arr_hour": 1},
        n_jobs=N_JOBS,
        **{"random_state": SEED, **c},
    )
    for c in CONFIGS
]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
for m in models:
    m.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
