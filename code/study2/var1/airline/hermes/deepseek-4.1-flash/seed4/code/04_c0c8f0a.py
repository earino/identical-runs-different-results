"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

feature_cols = [
    "DepTime", "tod", "hour", "minute", "tod_sin", "tod_cos",
    "Distance", "log_dist",
    "month_i", "dom_i", "dow_i",
] + RAW_CATS


def _ordinal(s: pd.Series) -> pd.Series:
    """'c-12' -> 12.0 for the c-<n> encoded columns."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").astype("float32")


cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CATS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    dep = df["DepTime"].astype("float32")
    hour = (dep // 100).clip(0, 24)
    minute = (dep % 100).clip(0, 59)
    tod = hour * 60 + minute
    dist = df["Distance"].astype("float32")
    m = _ordinal(df["Month"])
    d = _ordinal(df["DayofMonth"])
    w = _ordinal(df["DayOfWeek"])

    X = pd.DataFrame(index=df.index)
    X["DepTime"] = dep
    X["tod"] = tod
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["month_i"] = m
    X["dom_i"] = d
    X["dow_i"] = w
    for c in RAW_CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=40,
    reg_lambda=10.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
