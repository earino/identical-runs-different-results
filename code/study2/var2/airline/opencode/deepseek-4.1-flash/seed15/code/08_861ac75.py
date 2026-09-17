"""XGBoost binary classifier for airline departure delay. ONLY FILE THE AGENT EDITS.

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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba reproduces it on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (t // 100) % 24
    X["dep_hour"] = hour.astype("float32")
    X["dep_min"] = (t % 100).astype("float32")
    X["dep_abs"] = (hour * 60 + t % 100).astype("float32")
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["month_sin"] = np.sin(2 * np.pi * month / 12).astype("float32")
    X["month_cos"] = np.cos(2 * np.pi * month / 12).astype("float32")
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype("float32")
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype("float32")
    X["tod_sin"] = np.sin(2 * np.pi * X["dep_abs"] / 1440).astype("float32")
    X["tod_cos"] = np.cos(2 * np.pi * X["dep_abs"] / 1440).astype("float32")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    n_estimators=600,
    reg_lambda=5.0,
    reg_alpha=1.0,
)

MODEL_CONFIGS = [
    dict(max_depth=5, learning_rate=0.025, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42),
    dict(max_depth=6, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6, min_child_weight=10, random_state=7),
    dict(max_depth=4, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, random_state=2024),
    dict(max_depth=5, learning_rate=0.02, subsample=0.8, colsample_bytree=0.5, min_child_weight=8,
         n_estimators=900, random_state=123),
    dict(max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
         n_estimators=400, random_state=555),
]
Xtr_all = prepare(train)
ytr_all = to_y(train)


def _fit(cfg: dict) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier(**{**BASE, **cfg})
    m.fit(Xtr_all, ytr_all)
    return m


t0 = time.time()
models = [_fit(c) for c in MODEL_CONFIGS]
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
