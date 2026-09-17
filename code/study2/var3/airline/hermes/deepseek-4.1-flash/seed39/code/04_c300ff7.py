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

RAW_CATS = ["UniqueCarrier", "Origin", "Dest"]
# fit categorical levels on training data only
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CATS}
cat_cols = list(RAW_CATS)
cat_levels["tod_hour"] = pd.Index(np.arange(24))  # departure hour as a low-cardinality categorical
cat_levels["tod_half"] = pd.Index(np.arange(48))  # 30-minute departure bucket as categorical


def _codes(df, col):
    """c-<n> string -> integer n."""
    return df[col].astype(str).str.slice(2).astype(np.int16)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)

    # --- time of day (DepTime is hhmm) ---
    t = df["DepTime"].astype(np.int32)
    hh = (t // 100).to_numpy() % 24
    mm = np.minimum(t.to_numpy() % 100, 59)
    tod = hh * 60 + mm
    X["tod"] = tod.astype(np.int16)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["is_weekend"] = (df["DayOfWeek"].astype(str).str.slice(2).astype(np.int8).to_numpy() >= 6).astype(np.int8)
    X["tod_hour"] = pd.Categorical(hh, categories=cat_levels["tod_hour"])
    X["tod_half"] = pd.Categorical(tod // 30, categories=cat_levels["tod_half"])

    # --- calendar ---
    month = _codes(df, "Month")
    dom = _codes(df, "DayofMonth")
    dow = _codes(df, "DayOfWeek")
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["month_sin"] = np.sin(2 * np.pi * month.to_numpy() / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month.to_numpy() / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow.to_numpy() / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow.to_numpy() / 7.0)

    # --- distance ---
    dist = df["Distance"].astype(np.float32)
    X["log_dist"] = np.log1p(dist)

    # --- categoricals ---
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
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
