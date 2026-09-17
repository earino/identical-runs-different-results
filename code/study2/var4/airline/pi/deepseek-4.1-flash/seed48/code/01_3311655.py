"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

RAW_FEATURES = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
CAT_COLS = ["carrier", "origin", "dest", "route"]


def _cnum(s):
    return s.str.replace("c-", "", regex=False).astype(float)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic feature engineering (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["Distance"] = df["Distance"].astype(float)

    dep = df["DepTime"].astype(float)
    hour = dep // 100
    minute = dep % 100
    tod = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["tod"] = tod
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)

    mon = _cnum(df["Month"])
    dow = _cnum(df["DayOfWeek"])
    dom = _cnum(df["DayofMonth"])
    X["mon"] = mon
    X["dow"] = dow
    X["dom"] = dom
    X["is_weekend"] = (dow >= 6).astype(int)
    X["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    X["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    X["carrier"] = df["UniqueCarrier"].astype(str)
    X["origin"] = origin
    X["dest"] = dest
    X["route"] = origin + "_" + dest
    return X


_eng_train = engineer(train)
cat_levels = {c: pd.Index(sorted(_eng_train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = engineer(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
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
