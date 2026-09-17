"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
feature_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance",
                "hour", "minute", "tod_sin", "tod_cos",
                "month_sin", "month_cos", "dow_sin", "dow_cos",
                "log_dist"] + CAT_COLS
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = df["Month"].astype("string").str.slice(2).astype("float")
    dom = df["DayofMonth"].astype("string").str.slice(2).astype("float")
    dow = df["DayOfWeek"].astype("string").str.slice(2).astype("float")
    dt = pd.to_numeric(df["DepTime"], errors="coerce").astype("float")
    dt = dt.where(dt < 2400, dt - 2400)  # 2400 == midnight of the same slot -> 0
    hour = (dt // 100).fillna(-1)
    minute = (dt % 100).fillna(-1)
    tod = (hour * 60 + minute).where(dt.notna(), -1)
    X["Month"] = month
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["DepTime"] = dt
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype("float")
    X["log_dist"] = np.log1p(X["Distance"])
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440).astype("float")
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440).astype("float")
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_TREES = 200
PARAMS = {
    "max_depth": 16,
    "eta": 0.02,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "min_child_weight": 5.0,
    "lambda": 1.0,
    "tree_method": "hist",
    "nthread": N_JOBS,
}

t0 = time.time()
dtrain = xgb.DMatrix(prepare(train), label=to_y(train), enable_categorical=True)
deval = xgb.DMatrix(prepare(evald), enable_categorical=True)
models = []
for seed in (42, 7, 2026, 1, 13, 99):
    PARAMS["seed"] = seed
    models.append(xgb.train(PARAMS, dtrain, num_boost_round=N_TREES, verbose_eval=False))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dm = xgb.DMatrix(prepare(df), enable_categorical=True)
    ps = [np.asarray(m.predict(dm), dtype=float).ravel() for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
