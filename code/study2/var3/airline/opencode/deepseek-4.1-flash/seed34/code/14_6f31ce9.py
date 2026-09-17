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
ORD_CATS = ["Month", "DayofMonth", "DayOfWeek"]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000 and c not in ORD_CATS]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols or c in ORD_CATS]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
FREQ_CATS = ["Origin", "Dest", "UniqueCarrier"]
col_freq = {c: train[c].value_counts() for c in FREQ_CATS}
route_freq = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()

_tr_hour = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(lower=0, upper=23)
hour_freq = {}
for c in ["Origin", "Dest", "UniqueCarrier"]:
    hour_freq[c] = (train[c].astype(str) + "_" + _tr_hour.astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    route = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    X["route_freq"] = route.map(route_freq).fillna(0)
    for c in FREQ_CATS:
        X[c + "_freq"] = X[c].map(col_freq[c]).fillna(0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in ORD_CATS:
        X[c] = pd.to_numeric(X[c].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    for c, period in [("Month", 12.0), ("DayOfWeek", 7.0)]:
        X[c + "_sin"] = np.sin(2 * np.pi * X[c] / period)
        X[c + "_cos"] = np.cos(2 * np.pi * X[c] / period)
    dt = pd.to_numeric(X["DepTime"], errors="coerce")
    hour = (dt // 100).clip(lower=0, upper=23)
    tod = hour * 60 + (dt % 100).clip(lower=0, upper=59)
    X["dep_hour"] = hour
    X["dep_hour_cat"] = pd.Categorical(hour, categories=list(range(24)))
    X["dep_tod"] = tod
    X["dep_tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    for c in ["Origin", "Dest", "UniqueCarrier"]:
        key = X[c].astype(str) + "_" + hour.astype("int64").astype(str)
        X[c + "_hour_freq"] = key.map(hour_freq[c]).fillna(0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of shallow XGBoost models --------------------------
X_train = prepare(train)
y_train = to_y(train)

MODEL_SPECS = [
    (42, 3, 1.0),
    (7, 3, 0.9),
    (123, 4, 0.9),
    (2024, 3, 0.8),
    (555, 4, 1.0),
    (99, 3, 0.7),
    (314, 4, 0.8),
    (271, 3, 0.6),
    (13, 3, 0.9),
    (88, 4, 0.7),
    (409, 3, 0.8),
    (61, 4, 0.6),
]
models = []
t0 = time.time()
for seed, depth, colsample in MODEL_SPECS:
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=depth,
        learning_rate=0.05,
        colsample_bytree=colsample,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
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
