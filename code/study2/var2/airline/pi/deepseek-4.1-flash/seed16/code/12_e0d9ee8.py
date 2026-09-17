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


def _hr(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce") // 100).clip(0, 23).astype("Int64").astype(str)


# interaction categoricals: airport x hour-of-day and carrier x hour-of-day
oh_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + _hr(train["DepTime"])).unique()))
ch_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _hr(train["DepTime"])).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    minute = dep % 100
    X["dep_hour"] = hour
    X["dep_min"] = minute.where(minute < 60, 0)
    X["dep_since_midnight"] = (hour * 60).astype(float) + minute.where(minute < 60, 0).fillna(0)
    h = _hr(df["DepTime"])
    X["origin_hour"] = pd.Categorical(df["Origin"].astype(str) + "_" + h, categories=oh_levels)
    X["carrier_hour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + h, categories=ch_levels)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: average of diverse XGBoost models (variance reduction) -----------------
_BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
# two deep, heavily-regularized configs; seeds vary the column subsample (variance reduction)
_CFG_A = dict(n_estimators=700, max_depth=12, learning_rate=0.03, subsample=1.0, colsample_bytree=0.7,
              min_child_weight=5, reg_lambda=50.0, reg_alpha=0.5, gamma=0.0, max_cat_to_onehot=1, max_bin=512)
_CFG_B = dict(n_estimators=900, max_depth=12, learning_rate=0.03, subsample=1.0, colsample_bytree=0.5,
              min_child_weight=3, reg_lambda=30.0, reg_alpha=1.0, gamma=0.3, max_cat_to_onehot=4, max_bin=256)
_CFG_A2 = dict(_CFG_A, colsample_bylevel=0.7, colsample_bynode=0.7)
ENSEMBLE = [(_CFG_A, 42), (_CFG_A2, 42), (_CFG_B, 42)]

X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for params, seed in ENSEMBLE:
    m = xgb.XGBClassifier(random_state=seed, **_BASE, **params)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
