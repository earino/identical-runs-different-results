"""XGBoost binary classifier, experiment 5: time-of-day + calendar feature engineering.

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
# c-<n> string columns -> numeric; DepTime hhmm -> hour/minute/cyclic; log distance.
NUM_COLS = ["DepTime", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _cal_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _cal_num(df["Month"])
    dom = _cal_num(df["DayofMonth"])
    dow = _cal_num(df["DayOfWeek"])
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 29)
    minute = dt % 100
    mins = (hour * 60 + minute).clip(0, 24 * 60 + 59)

    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["DepTime"] = dt
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_dist"] = np.log1p(X["Distance"])
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["doy"] = month * 31 + dom  # rough seasonality proxy
    X["doy_sin"] = np.sin(2 * np.pi * X["doy"] / 372.0)
    X["doy_cos"] = np.cos(2 * np.pi * X["doy"] / 372.0)
    X["is_weekend"] = (dow >= 6).astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


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
