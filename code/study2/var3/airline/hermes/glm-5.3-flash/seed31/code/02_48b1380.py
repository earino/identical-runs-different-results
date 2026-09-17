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
N_TREES = 4000
PARAMS = {
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "auc",
    "seed": SEED,
    "nthread": N_JOBS,
}

val = train.iloc[:20000]      # early-2005 rows: a time-respecting probe for tuning/early stopping
full = train.iloc[20000:]

t0 = time.time()
dprobe = xgb.DMatrix(prepare(val), label=to_y(val), enable_categorical=True)
dtrain = xgb.DMatrix(prepare(full), label=to_y(full), enable_categorical=True)
evh = [(dprobe, "v")]
probe = xgb.train(PARAMS, dtrain, num_boost_round=N_TREES, evals=evh,
                  early_stopping_rounds=80, verbose_eval=False)
best_it = probe.best_iteration
print(f"Training time: {time.time() - t0:.1f}s")
print(f"best_iteration on time-respecting probe: {best_it}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    p = probe.predict(xgb.DMatrix(prepare(df), enable_categorical=True))
    return np.asarray(p, dtype=float).ravel()


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
