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
    X["dep_hour"] = (df["DepTime"].astype(float) // 100) % 24
    X["dep_hour"] = pd.Categorical(X["dep_hour"], categories=pd.Index(range(24)))
    X["dep_qtrhour"] = pd.Categorical(
        (df["DepTime"].astype(float) // 15) % 96,
        categories=pd.Index(range(96)),
    )
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
DEPTH_LADDER = [2, 3, 4, 5, 6]
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
month_num = train["Month"].str[2:].astype(int).to_numpy()
wtr = 1.0 + (month_num - 1) / 11.0  # recency weights: later 2005 months count more
models = []
LRS = [0.05, 0.1, 0.2]
ALPHAS = [0.5, 1.0, 2.0]
for d in DEPTH_LADDER:
  for lr in LRS:
    for alpha in ALPHAS:
        m = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=d,
            learning_rate=lr,
            min_child_weight=20,
            colsample_bytree=0.7,
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED + d * 10 + int(lr * 100) + int(alpha * 10),
            n_jobs=N_JOBS,
        )
        m.fit(Xtr, ytr, sample_weight=1.0 + alpha * (month_num - 1) / 11.0)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  models: {len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
