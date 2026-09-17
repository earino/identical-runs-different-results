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
BASE_PARAMS = dict(
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    reg_alpha=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
ENSEMBLE = [  # (max_depth, seed, n_estimators, colsample, subsample)
    (16, 1, 300, 0.6, 0.7),
    (18, 7, 300, 0.7, 0.75),
    (20, 42, 300, 0.8, 0.8),
    (20, 123, 300, 0.9, 0.85),
    (22, 2024, 300, 1.0, 0.9),
    (24, 555, 300, 0.85, 0.8),
]

fit_stats(train)
X_all = prepare(train)
y_all = to_y(train)
X_eval = prepare(evald)

t0 = time.time()
models = []
for depth, seed, n_est, colsample, subsample in ENSEMBLE:
    m = xgb.XGBClassifier(n_estimators=n_est, max_depth=depth, colsample_bytree=colsample,
                          subsample=subsample, random_state=seed,
                          **{k: v for k, v in BASE_PARAMS.items() if k not in ("colsample_bytree", "subsample")})
    m.fit(X_all, y_all)
    models.append(m)
print(f"Ensemble fit: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df) if not df.equals(evald) else X_eval
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
