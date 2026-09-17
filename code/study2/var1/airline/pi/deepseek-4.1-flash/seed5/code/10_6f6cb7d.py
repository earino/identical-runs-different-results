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
ohe_cols = [f"{c}_{v}" for c in cat_cols for v in cat_levels[c]]

# Congestion proxies: flights per airport per scheduled hour (fit on TRAINING data only).
_tr_hour = (train["DepTime"] // 100).clip(0, 23)
_origin_hour = train["Origin"].astype(str) + "_" + _tr_hour.astype(str)
_dest_hour = train["Dest"].astype(str) + "_" + _tr_hour.astype(str)
origin_hour_counts = _origin_hour.value_counts()
dest_hour_counts = _dest_hour.value_counts()
origin_totals = train["Origin"].value_counts()
dest_totals = train["Dest"].value_counts()
# Estimated arrival hour (distance / ~500 mph) -> destination arrival congestion.
_tr_arr_hour = (_tr_hour + (train["Distance"] / 500.0).round()).clip(0, 23).astype(int)
dest_arr_hour_counts = (train["Dest"].astype(str) + "_" + _tr_arr_hour.astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    hour = (pd.to_numeric(X["DepTime"], errors="coerce") // 100).clip(0, 23)
    X["origin_hour_cnt"] = (df["Origin"].astype(str) + "_" + hour.astype(str)).map(origin_hour_counts).fillna(0)
    X["dest_hour_cnt"] = (df["Dest"].astype(str) + "_" + hour.astype(str)).map(dest_hour_counts).fillna(0)
    X["origin_hour_ratio"] = X["origin_hour_cnt"] / df["Origin"].astype(str).map(origin_totals).fillna(1)
    X["dest_hour_ratio"] = X["dest_hour_cnt"] / df["Dest"].astype(str).map(dest_totals).fillna(1)
    X["dep_hour"] = hour
    arr_hour = (hour + (pd.to_numeric(df["Distance"], errors="coerce") / 500.0).round()).clip(0, 23).astype(int)
    X["dest_arr_hour_ratio"] = (df["Dest"].astype(str) + "_" + arr_hour.astype(str)).map(dest_arr_hour_counts).fillna(0) / df["Dest"].astype(str).map(dest_totals).fillna(1)
    X["origin_inbound_hour_cnt"] = (df["Origin"].astype(str) + "_" + hour.astype(str)).map(dest_arr_hour_counts).fillna(0)
    X["origin_inbound_ratio"] = X["origin_inbound_hour_cnt"] / df["Origin"].astype(str).map(origin_totals).fillna(1)
    X["dest_outbound_hour_cnt"] = (df["Dest"].astype(str) + "_" + hour.astype(str)).map(origin_hour_counts).fillna(0)
    X["dest_outbound_ratio"] = X["dest_outbound_hour_cnt"] / df["Dest"].astype(str).map(dest_totals).fillna(1)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    d = pd.get_dummies(X[cat_cols], columns=cat_cols, dtype=np.int8)
    d = d.reindex(columns=ohe_cols, fill_value=0)
    X = pd.concat([X.drop(columns=cat_cols), d], axis=1)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Ensemble of regularized XGBoost models; averaging reduces variance under 2005->2006 drift.
MODEL_PARAMS = [
    dict(n_estimators=280, max_depth=24, learning_rate=0.05, min_child_weight=1,
         subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0, random_state=42),
    dict(n_estimators=280, max_depth=24, learning_rate=0.05, min_child_weight=1,
         subsample=0.9, colsample_bytree=0.8, reg_lambda=5.0, random_state=7),
    dict(n_estimators=160, max_depth=36, learning_rate=0.05, min_child_weight=1,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, random_state=11),
]

models = [
    xgb.XGBClassifier(tree_method="hist", n_jobs=N_JOBS, **p)
    for p in MODEL_PARAMS
]

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
