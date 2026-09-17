"""XGBoost binary classifier on the airline delay dataset. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_eval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

# --- features -----------------------------------------------------------------
TIME_COLS = ["Month", "DayofMonth", "DayOfWeek"]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse c-N strings to ints and add cyclical encodings + DepTime components."""
    for c in TIME_COLS:
        df[c + "_n"] = df[c].str[2:].astype(float)
    m = df["Month_n"]
    dom = df["DayofMonth_n"]
    dow = df["DayOfWeek_n"]
    dt = df["DepTime"].astype(float)
    hour = np.floor(dt / 100.0).clip(0, 23)
    minute = dt - np.floor(dt / 100.0) * 100
    tmin = hour * 60 + minute
    df["dep_hour"] = hour
    df["dep_t_sin"] = np.sin(2 * np.pi * tmin / 1440)
    df["dep_t_cos"] = np.cos(2 * np.pi * tmin / 1440)
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    df["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["dist_log"] = np.log1p(df["Distance"].astype(float))
    return df


NUM_COLS = [c + "_n" for c in TIME_COLS] + [
    "dep_hour", "dep_t_sin", "dep_t_cos",
    "month_sin", "month_cos", "dom_sin", "dom_cos", "dow_sin", "dow_cos",
    "Distance", "dist_log",
]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    df = df.copy()
    add_time_features(df)
    X = df[NUM_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train), y_train,
    eval_set=[(prepare(evald), y_eval)],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s (best_iteration={model.best_iteration})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
