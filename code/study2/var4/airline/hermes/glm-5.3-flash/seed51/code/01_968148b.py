"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
NUM_COLS = [
    "Month", "DayofMonth", "DayOfWeek",
    "dep_hour", "dep_min", "dep_sin", "dep_cos", "dep_wrap",
    "Distance", "log_distance",
]
FEATURE_COLS = CAT_COLS + NUM_COLS
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    month = df["Month"].astype(str).str.slice(2).astype(int)
    dom = df["DayofMonth"].astype(str).str.slice(2).astype(int)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(int)
    dt = df["DepTime"].astype(int)
    hour = (dt // 100) % 24
    minute = dt % 100
    wrap = (dt >= 2400).astype(int)
    ang = 2.0 * np.pi * (hour * 60.0 + minute) / 1440.0

    X["Month"] = month
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["dep_wrap"] = wrap
    X["Distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["Distance"])
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=10.0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train), to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s (trees={model.best_iteration + 1})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
