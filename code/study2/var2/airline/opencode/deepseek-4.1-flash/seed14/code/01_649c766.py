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

# --- features -----------------------------------------------------------------
# c-<n> encoded categorical columns (Month, DayofMonth, DayOfWeek): decode to int.
CYCLIC = {"Month": 12, "DayofMonth": 31, "DayOfWeek": 7}
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}


def _cint(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CYCLIC:
        v = _cint(df[c])
        X[c] = v
        period = CYCLIC[c]
        X[f"{c}_sin"] = np.sin(2 * np.pi * v / period)
        X[f"{c}_cos"] = np.cos(2 * np.pi * v / period)
    # DepTime is hhmm; values >2400 are after midnight.
    dep = df["DepTime"].astype(float)
    hour = (dep // 100).astype(int) % 24
    minute = (dep % 100).astype(int)
    tod = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_tod"] = tod
    X["dep_tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
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
