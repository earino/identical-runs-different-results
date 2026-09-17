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
_o_hour = (
    train["Origin"].astype(str) + "_" + (pd.to_numeric(train["DepTime"], errors="coerce") // 100).astype("Int64").astype(str)
).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hour = (pd.to_numeric(df["DepTime"], errors="coerce") // 100).astype("Int64")
    X["f_origin_hour"] = (df["Origin"].astype(str) + "_" + hour.astype(str)).map(_o_hour).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, seed=1),
    dict(n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, seed=2),
    dict(n_estimators=400, max_depth=2, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, seed=3),
    dict(n_estimators=300, max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, seed=4),
    dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, seed=5),
    dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7, seed=6),
    dict(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, seed=7),
    dict(n_estimators=250, max_depth=3, learning_rate=0.1, subsample=1.0, colsample_bytree=0.5, seed=8),
    dict(n_estimators=200, max_depth=4, learning_rate=0.07, subsample=0.8, colsample_bytree=0.6, seed=9),
    dict(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=1.0, colsample_bytree=1.0, seed=10),
]

X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg["seed"],
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
