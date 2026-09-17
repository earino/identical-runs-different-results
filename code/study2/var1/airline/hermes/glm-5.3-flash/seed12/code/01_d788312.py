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

# --- feature engineering ------------------------------------------------------
# categorical columns kept as native categoricals
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
# derived numeric features (built in prepare())
DERIVED = [
    "DepHour", "DepMinute", "DepMinSin", "DepMinCos", "DepTimeBin",
    "MonthN", "DayOfMonthN", "DayOfWeekN", "MonthSin", "MonthCos",
    "DayOfWeekSin", "DayOfWeekCos", "LogDistance", "Weekend",
]
TIME_BINS = ["night", "early_morning", "morning", "midday", "afternoon", "evening"]

FEATURE_COLS = CAT_COLS + DERIVED


def _cnum(s: pd.Series) -> pd.Series:
    """c-<n> string -> integer n."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _tod_bin(hour: pd.Series) -> pd.Series:
    b = pd.cut(hour, bins=[-1, 4, 7, 10, 15, 19, 23, 24], labels=TIME_BINS + ["late_night"])
    return b.astype(str).where(hour.notna(), "missing")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN

    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 23)
    minute = (dt % 100).clip(0, 59)
    frac = (hour * 60 + minute) / 1440.0
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["DepMinSin"] = np.sin(2 * np.pi * frac)
    X["DepMinCos"] = np.cos(2 * np.pi * frac)
    X["DepTimeBin"] = pd.Categorical(_tod_bin(hour), categories=TIME_BINS + ["late_night"])

    month = _cnum(df["Month"]).clip(1, 12)
    dom = _cnum(df["DayofMonth"]).clip(1, 31)
    dow = _cnum(df["DayOfWeek"]).clip(1, 7)
    X["MonthN"] = month
    X["DayOfMonthN"] = dom
    X["DayOfWeekN"] = dow
    X["MonthSin"] = np.sin(2 * np.pi * month / 12)
    X["MonthCos"] = np.cos(2 * np.pi * month / 12)
    X["DayOfWeekSin"] = np.sin(2 * np.pi * dow / 7)
    X["DayOfWeekCos"] = np.cos(2 * np.pi * dow / 7)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDistance"] = np.log1p(dist)

    X["Weekend"] = (dow >= 6).astype(float)
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
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
