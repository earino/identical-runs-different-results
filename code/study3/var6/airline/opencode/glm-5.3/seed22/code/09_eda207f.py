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
num_cols = [c for c in feature_cols if c not in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[num_cols].copy()
    # calendar strings -> ordinal ints
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c + "_n"] = df[c].str.replace("c-", "", regex=False).astype(int)
    # scheduled departure time -> hour / minute / minutes-of-day + cyclic encoding
    dep = df["DepTime"].astype(int)
    hour, minute = dep // 100, dep % 100
    tod = (hour * 60 + minute).astype(float)
    X["hour"] = hour
    X["tod"] = tod
    frac = tod / 1440.0
    X["tod_sin"] = np.sin(2 * np.pi * frac)
    X["tod_cos"] = np.cos(2 * np.pi * frac)
    # categorical features
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    n_estimators=8000,
    learning_rate=0.03,
    min_child_weight=100,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    early_stopping_rounds=150,
    eval_metric="auc",
)
VARIANTS = [
    dict(max_depth=8, subsample=0.8, colsample_bytree=0.7, random_state=42),
    dict(max_depth=10, subsample=0.7, colsample_bytree=0.6, random_state=42),
    dict(max_depth=10, subsample=0.75, colsample_bytree=0.5, random_state=7),
    dict(max_depth=12, subsample=0.7, colsample_bytree=0.6, random_state=2024),
    dict(max_depth=12, subsample=0.6, colsample_bytree=0.7, random_state=1),
    dict(max_depth=20, max_leaves=64, grow_policy="lossguide", subsample=0.7, colsample_bytree=0.6, random_state=99),
    dict(max_depth=20, max_leaves=128, grow_policy="lossguide", subsample=0.8, colsample_bytree=0.7, random_state=123),
    dict(max_depth=20, max_leaves=256, grow_policy="lossguide", subsample=0.6, colsample_bytree=0.5, random_state=5),
]

X_tr, y_tr = prepare(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)

t0 = time.time()
models = []
for var in VARIANTS:
    m = xgb.XGBClassifier(**BASE, **var)
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    models.append(m)
    print(f"  {var}: best_iter={m.best_iteration} auc={m.best_score:.5f}")
model = models[0]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
