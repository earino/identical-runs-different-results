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
ORDINAL_COLS = ["Month", "DayofMonth", "DayOfWeek"]
cat_cols = [c for c in cat_cols if c not in ORDINAL_COLS]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
feature_cols = [c for c in feature_cols if c != "DepTime"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
freq_origin = train["Origin"].value_counts()
freq_dest = train["Dest"].value_counts()
freq_carrier = train["UniqueCarrier"].value_counts()
freq_route = route_tr.value_counts()
freq_carrier_route = (train["UniqueCarrier"].astype(str) + "_" + route_tr).value_counts()
freq_origin_hour = (train["Origin"].astype(str) + "_" + (train["DepTime"].astype(int) // 100).astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in ORDINAL_COLS:
        X[c] = df[c].astype(str).str.replace("c-", "", regex=False).astype(int)
    dep = df["DepTime"].astype(float)
    dep = np.where(dep >= 2400, dep - 2400, dep)
    hours = np.floor(dep / 100.0)
    dep_min = hours * 60.0 + (dep - hours * 100.0)
    X["dep_min"] = dep_min
    X["dep_hour"] = hours
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440.0)
    m = X["Month"].to_numpy()
    X["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    X["freq_origin"] = origin.map(freq_origin).fillna(0)
    X["freq_dest"] = dest.map(freq_dest).fillna(0)
    X["freq_carrier"] = df["UniqueCarrier"].astype(str).map(freq_carrier).fillna(0)
    X["freq_route"] = (origin + "_" + dest).map(freq_route).fillna(0)
    X["freq_carrier_route"] = (df["UniqueCarrier"].astype(str) + "_" + origin + "_" + dest).map(freq_carrier_route).fillna(0)
    X["freq_origin_hour"] = (origin + "_" + X["dep_hour"].astype(int).astype(str)).map(freq_origin_hour).fillna(0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
ENSEMBLE = [
    dict(n_estimators=400, max_depth=10, learning_rate=0.03, min_child_weight=25, subsample=0.7, colsample_bytree=0.7),
    dict(n_estimators=350, max_depth=12, learning_rate=0.03, min_child_weight=40, subsample=0.7, colsample_bytree=0.7),
    dict(n_estimators=300, max_depth=14, learning_rate=0.03, min_child_weight=60, subsample=0.7, colsample_bytree=0.7),
    dict(n_estimators=500, max_depth=10, learning_rate=0.02, min_child_weight=25, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=300, max_depth=12, learning_rate=0.04, min_child_weight=40, subsample=0.75, colsample_bytree=0.75),
]
X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for i, params in enumerate(ENSEMBLE):
    m = xgb.XGBClassifier(**BASE, random_state=SEED + i, **params)
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
