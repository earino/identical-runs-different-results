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
# frequency maps fitted on training data only
origin_freq = train["Origin"].value_counts(normalize=True)
dest_freq = train["Dest"].value_counts(normalize=True)
carrier_freq = train["UniqueCarrier"].value_counts(normalize=True)
route_freq = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts(normalize=True)
_tr_hour = pd.to_numeric(train["DepTime"], errors="coerce") // 100
origin_hour_freq = (train["Origin"].astype(str) + "_" + _tr_hour.astype(str)).value_counts(normalize=True)
dest_hour_freq = (train["Dest"].astype(str) + "_" + _tr_hour.astype(str)).value_counts(normalize=True)
carrier_hour_freq = (train["UniqueCarrier"].astype(str) + "_" + _tr_hour.astype(str)).value_counts(normalize=True)


def _cat_num(s: pd.Series) -> pd.Series:
    # "c-4" -> 4
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    drop = {"DepTime", "Distance", "DayofMonth"}
    keep = [c for c in feature_cols if c not in drop]
    X = df[keep].copy()
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = dep // 100
    minute = dep % 100
    mins = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["dep_mins"] = mins
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["log_distance"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    X["origin_freq"] = df["Origin"].map(origin_freq).fillna(0.0)
    X["dest_freq"] = df["Dest"].map(dest_freq).fillna(0.0)
    X["carrier_freq"] = df["UniqueCarrier"].map(carrier_freq).fillna(0.0)
    X["airport_freq"] = X["origin_freq"] + X["dest_freq"]
    X["route_freq"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(route_freq).fillna(0.0)
    X["origin_hour_freq"] = (df["Origin"].astype(str) + "_" + hour.astype(str)).map(origin_hour_freq).fillna(0.0)
    X["dest_hour_freq"] = (df["Dest"].astype(str) + "_" + hour.astype(str)).map(dest_hour_freq).fillna(0.0)
    X["carrier_hour_freq"] = (df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)).map(carrier_hour_freq).fillna(0.0)
    X["origin_hour_share"] = X["origin_hour_freq"] / X["origin_freq"].replace(0, np.nan)
    X["dest_hour_share"] = X["dest_hour_freq"] / X["dest_freq"].replace(0, np.nan)
    X["carrier_hour_share"] = X["carrier_hour_freq"] / X["carrier_freq"].replace(0, np.nan)
    X["route_hour_share"] = X["origin_hour_freq"] / X["route_freq"].replace(0, np.nan)
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    n_estimators=1000,
    learning_rate=0.03,
    subsample=0.8,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
PARAMS = [
    dict(max_depth=11, colsample_bytree=0.4, min_child_weight=80, reg_lambda=10.0, reg_alpha=5.0, random_state=42),
    dict(max_depth=12, colsample_bytree=0.4, min_child_weight=80, reg_lambda=10.0, reg_alpha=5.0, random_state=101),
    dict(max_depth=13, colsample_bytree=0.35, min_child_weight=100, reg_lambda=12.0, reg_alpha=6.0, random_state=202),
]

models = []
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
for p in PARAMS:
    m = xgb.XGBClassifier(**BASE, **p)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
