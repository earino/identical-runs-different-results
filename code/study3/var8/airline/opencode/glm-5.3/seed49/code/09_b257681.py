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
cat_cols = [c for c in cat_cols if c != "DayofMonth"]  # ablation: day-of-month is 2005 noise
feature_cols = [c for c in feature_cols if c != "DayofMonth"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of the baseline config with (sub, col)-sample diversity -
N_SEEDS = 120
GRID = [(d, s, c) for d in (5, 6, 7, 8) for s in (0.7, 0.8, 0.9) for c in (0.7, 0.8, 0.9)]  # 36 configs
t0 = time.time()
y_train = to_y(train)
Xtr = prepare(train)
# recency: ramp weights through 2005 (Jan 1.0 -> Dec 2.0), temporally closest to the 2006 eval year
_month_n = pd.to_numeric(train["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce").to_numpy()
w_train = 1.0 + (np.clip(_month_n, 1, 12) - 1) / 11.0
models = []
for i in range(N_SEEDS):
    d, s, c = GRID[i % len(GRID)]
    m = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=d,
        learning_rate=0.1,
        subsample=s,
        colsample_bytree=c,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 10 * i,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, y_train, sample_weight=w_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_SEEDS} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
