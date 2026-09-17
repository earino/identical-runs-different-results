"""XGBoost binary classifier for the airline delay task.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare(); encoders/statistics are fit on the training data only.
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

# --- feature specification (fit on training data only) ------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    dep = df["DepTime"].astype("float64")
    X["hour"] = dep // 100.0
    X["minute"] = dep % 100.0
    X["minute_of_day"] = dep
    X["log_distance"] = np.log1p(df["Distance"].astype("float64"))
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24.0)
    # shift the day to start at 04:00, where delay risk is lowest and (mostly) monotone in time
    shifted = ((X["hour"] - 4.0) % 24.0) * 60.0 + X["minute"]
    X["shifted_minute"] = shifted
    X["shifted_sin"] = np.sin(2 * np.pi * shifted / 1440.0)
    X["shifted_cos"] = np.cos(2 * np.pi * shifted / 1440.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(n_estimators=1200, learning_rate=0.02, tree_method="hist",
            enable_categorical=True, eval_metric="auc", n_jobs=N_JOBS)
PARAMS = [
    dict(max_depth=6, reg_alpha=10.0, reg_lambda=20.0, min_child_weight=10,
         subsample=0.7, colsample_bytree=0.6, random_state=42),
    dict(max_depth=6, reg_alpha=10.0, reg_lambda=20.0, min_child_weight=10,
         subsample=0.7, colsample_bytree=0.6, random_state=101),
    dict(max_depth=7, reg_alpha=15.0, reg_lambda=30.0, min_child_weight=20,
         subsample=0.7, colsample_bytree=0.5, random_state=202),
    dict(max_depth=5, reg_alpha=8.0, reg_lambda=15.0, min_child_weight=5,
         subsample=0.8, colsample_bytree=0.7, random_state=303),
]

X_train, y_train = prepare(train), to_y(train)
t0 = time.time()
models = []
for p in PARAMS:
    m = xgb.XGBClassifier(**BASE, **p)
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
