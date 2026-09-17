"""XGBoost airline-delay classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
hour_levels = [str(i) for i in range(25)]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # cyclic day-of-year from Month/DayofMonth (c-<n> strings)
    month = df["Month"].str.slice(2).astype(float)
    day = df["DayofMonth"].str.slice(2).astype(float)
    doy = (month - 1.0) * 30.4 + day
    X["SeasonSin"] = np.sin(2 * np.pi * doy / 365.0)
    X["SeasonCos"] = np.cos(2 * np.pi * doy / 365.0)
    # hour-of-day as categorical (delay risk is strongly time-of-day shaped)
    hour = np.floor(df["DepTime"].astype(float) / 100.0).clip(0, 24).astype(int).astype(str)
    X["DepHour"] = pd.Categorical(hour, categories=hour_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# small ensemble of XGBoost models with different seeds; averaging reduces variance
SEEDS = [42, 7, 2024]
models = [
    xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    for s in SEEDS
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
