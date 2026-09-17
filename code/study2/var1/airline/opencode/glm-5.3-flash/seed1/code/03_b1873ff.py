"""XGBoost binary classifier on the airline delay dataset.

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

FEATURE_COLS = [
    "Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier",
    "Origin", "Dest", "Distance",
    "month_sin", "month_cos", "dom_sin", "dom_cos", "dow_sin", "dow_cos",
    "tod_sin", "tod_cos", "hour", "hour_cat", "log_dist",
]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "hour_cat"]
RAW_CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
HOUR_CATS = [str(i) for i in range(24)]


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(int)


def fit_stats(df: pd.DataFrame) -> None:
    """Fit encoders/statistics on training data only."""
    global CAT_LEVELS
    CAT_LEVELS = {c: pd.Index(sorted(df[c].astype(str).unique())) for c in RAW_CAT_COLS}
    CAT_LEVELS["hour_cat"] = pd.Index(HOUR_CATS)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    deptime = df["DepTime"].astype(float)
    hour = deptime // 100
    minute = deptime % 100
    tod = (hour + minute / 60.0) % 24.0
    ang = 2 * np.pi / 24.0
    X["Month"] = month
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["DepTime"] = deptime
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["tod_sin"] = np.sin(ang * tod)
    X["tod_cos"] = np.cos(ang * tod)
    X["hour"] = hour % 24
    X["hour_cat"] = pd.Categorical((hour % 24).astype(int).astype(str), categories=[str(i) for i in range(24)])
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

fit_stats(train)

t0 = time.time()
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = len(train) // 10
val_idx, tr_idx = idx[:n_val], idx[n_val:]
model = xgb.XGBClassifier(early_stopping_rounds=50, eval_metric="auc", **PARAMS)
model.fit(
    prepare(train.iloc[tr_idx]), to_y(train.iloc[tr_idx]),
    eval_set=[(prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx]))],
    verbose=False,
)
best_ntree = model.best_iteration + 1
print(f"ES fit: {best_ntree} rounds in {time.time() - t0:.1f}s")

t0 = time.time()
model = xgb.XGBClassifier(n_estimators=best_ntree, **{k: v for k, v in PARAMS.items() if k != "n_estimators"})
model.fit(prepare(train), to_y(train))
print(f"Full fit: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
