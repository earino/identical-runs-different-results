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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


HOUR_LEVELS = sorted((train["DepTime"] // 100).unique())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    X["minute"] = (df["DepTime"] % 100).astype("int16")
    X["hour"] = pd.Categorical((df["DepTime"] // 100).astype(int), categories=HOUR_LEVELS)
    X["dep_minutes"] = ((df["DepTime"] // 100) * 60 + df["DepTime"] % 100).astype("int32")
    X["min_bin"] = ((df["DepTime"] % 100) // 5).astype("int8")
    X["log_dist"] = np.log1p(df["Distance"]).astype("float32")
    X["dom_bin"] = (df["DayofMonth"].str[2:].astype(int) - 1) // 10
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Seed-bagged XGBoost ensemble: same hyperparameters, different random_state
# (subsampling rows/columns), predictions averaged. Reduces variance.
common = dict(
    n_estimators=200,
    max_depth=20,
    learning_rate=0.05,
    min_child_weight=10,
    reg_lambda=10,
    subsample=0.7,
    colsample_bytree=0.4,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

models = []
t0 = time.time()
for seed in (42, 7, 123, 2024, 99):
    m = xgb.XGBClassifier(random_state=seed, **common)
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
