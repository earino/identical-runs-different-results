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
    # time parsing: Month/DayofMonth/DayOfWeek are c-<n> strings, DepTime is hhmm
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        if c in X.columns:
            X[c] = X[c].astype(str).str.slice(2).astype("float32")
    if "DepTime" in X.columns:
        dt = pd.to_numeric(X["DepTime"], errors="coerce")
        dep_min = (dt // 100 * 60 + dt % 100) % 1440
        X["dep_min"] = dep_min.astype("float32")
        X["dep_hour"] = (dep_min // 60).astype("float32")
        X["dep_minute"] = (dep_min % 60).astype("float32")
        X.drop(columns=["DepTime"], inplace=True)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# n_estimators tuned by early stopping on an internal split (best_iter~523); fixed here so
# both bagging members fit on the full training set within the time budget.
N_TREES = 523
X_full, y_full = prepare(train), to_y(train)

CONFIGS = [
    dict(max_depth=20, learning_rate=0.01, min_child_weight=3, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, random_state=42),
    dict(max_depth=20, learning_rate=0.01, min_child_weight=3, subsample=0.7, colsample_bytree=0.5, reg_lambda=1.0, random_state=7),
]

models = []
for cfg in CONFIGS:
    m = xgb.XGBClassifier(n_estimators=N_TREES, tree_method="hist", enable_categorical=True, max_bin=2048,
                          n_jobs=N_JOBS, **cfg)
    t0 = time.time()
    m.fit(X_full, y_full)
    print(f"cfg=seed{cfg['random_state']}/cols{cfg['colsample_bytree']} fit {time.time() - t0:.1f}s")
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
