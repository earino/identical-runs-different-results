"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features ------------------------------------------------------------------
# `c-<n>` calendar columns are integer valued but stored as strings -> decode to ints.
C_NUM_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba() reproduces it on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in C_NUM_COLS:
        X[c] = df[c].str.replace("c-", "", regex=False).astype(np.int16)
    for c in NUM_COLS:
        X[c] = df[c].astype(np.float32)

    dep = df["DepTime"].astype(np.int32)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["DepHour"] = hour.astype(np.int16)
    X["DepMinOfDay"] = (hour * 60 + minute).astype(np.int16)

    X["HourSin"] = np.sin(2 * np.pi * X["DepHour"] / 24).astype(np.float32)
    X["HourCos"] = np.cos(2 * np.pi * X["DepHour"] / 24).astype(np.float32)
    doy = (X["Month"] - 1) * 30 + X["DayofMonth"]
    X["DoySin"] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    X["DoyCos"] = np.cos(2 * np.pi * doy / 365).astype(np.float32)

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ---------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=30,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_lambda=5.0,
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
