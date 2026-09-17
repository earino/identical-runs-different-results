"""XGBoost binary classifier for flight-delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# columns whose values are string codes of the form "c-<int>"
CODE_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _decode(series: pd.Series) -> pd.Series:
    """'c-11' -> 11.0 (NaN when the value is missing or malformed)."""
    return pd.to_numeric(series.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CODE_COLS:
        X[c] = _decode(df[c])
    # DepTime is scheduled departure as hhmm; decompose into time-of-day position (cycle-safe sin/cos).
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(lower=0, upper=28)
    minute = dep % 100
    tod = (hour * 60 + minute).clip(lower=0, upper=28 * 60 + 59)
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    # Distance: raw + log, plus a coarse flight-length band.
    dist = pd.to_numeric(df["Distance"], errors="coerce").clip(lower=1)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=4,
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
