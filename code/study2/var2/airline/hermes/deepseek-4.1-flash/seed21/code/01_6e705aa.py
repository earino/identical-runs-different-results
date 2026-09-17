"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives in `prepare()`, which is the single code path used both for fitting and for
scoring unseen rows (including the hidden holdout).
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

NUM_FEATS = ["hour", "minute", "tod", "tod_sin", "tod_cos", "Distance", "log_dist",
             "Month_n", "Day_n", "Dow_n", "is_weekend"]
CAT_FEATS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_FEATS}
HOUR_LEVELS = pd.Index([str(i) for i in range(24)])
TARGET_RATE = 0.5


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> model matrix. Used for training AND for predict_proba on unseen rows."""
    dt = df["DepTime"].astype(int)
    hour = (dt // 100) % 24
    minute = (dt % 100).clip(0, 59)
    tod = hour + minute / 60.0
    X = pd.DataFrame(index=df.index)
    # scheduled departure time: hour as a categorical captures the strong, stable intraday delay ramp
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24)
    # distance
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    # calendar, as ordered numbers (c-<n> strings)
    month = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dom = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dow = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["Month_n"] = month
    X["Day_n"] = dom
    X["Dow_n"] = dow
    X["is_weekend"] = (dow >= 6).astype(int)
    X["hour_cat"] = pd.Categorical(hour.astype(str), categories=HOUR_LEVELS)
    for c in CAT_FEATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.05,
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
