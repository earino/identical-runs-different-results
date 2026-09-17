"""XGBoost binary classifier for airline departure delay (autoresearch experiment).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare() so it also applies to unseen rows.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# categorical levels are learned from TRAIN only; unseen levels -> NaN for xgboost
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num(s):
    return pd.to_numeric(s.astype(str).str[2:], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature is derived here from the raw columns, so predict_proba() can reproduce it."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    # time-of-day: scheduled departure hhmm -> hour / minute-of-day / cyclic encoding
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    h = (dep // 100).clip(0, 23)
    m = (dep % 100).clip(0, 59)
    tod = h + m / 60.0
    X["hour"] = h
    X["minofday"] = h * 60 + m
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)

    # calendar cycles (Month/DayOfWeek arrive as c-<n> strings)
    mo = _num(df["Month"])
    dw = _num(df["DayOfWeek"])
    X["mo_sin"] = np.sin(2 * np.pi * mo / 12.0)
    X["mo_cos"] = np.cos(2 * np.pi * mo / 12.0)
    X["dw_sin"] = np.sin(2 * np.pi * dw / 7.0)
    X["dw_cos"] = np.cos(2 * np.pi * dw / 7.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=3,
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
