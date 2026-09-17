"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[["DepTime", "Distance"]].copy()
    dep = df["DepTime"].astype(int)
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["logdist"] = np.log(df["Distance"].astype(float))
    X["night"] = (((dep // 100) <= 4) | ((dep // 100) >= 24)).astype(int)
    ang = 2 * np.pi * ((dep // 100) * 60 + dep % 100) / 1440.0
    X["sin_t"] = np.sin(ang)
    X["cos_t"] = np.cos(ang)
    X["sin2_t"] = np.sin(2 * ang)
    X["cos2_t"] = np.cos(2 * ang)
    X["sin3_t"] = np.sin(3 * ang)
    X["cos3_t"] = np.cos(3 * ang)
    X["sin4_t"] = np.sin(4 * ang)
    X["cos4_t"] = np.cos(4 * ang)
    X["sin5_t"] = np.sin(5 * ang)
    X["cos5_t"] = np.cos(5 * ang)
    X["sin6_t"] = np.sin(6 * ang)
    X["cos6_t"] = np.cos(6 * ang)
    X["sin7_t"] = np.sin(7 * ang)
    X["cos7_t"] = np.cos(7 * ang)
    X["sin8_t"] = np.sin(8 * ang)
    X["cos8_t"] = np.cos(8 * ang)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# 2005->2006 drift punishes single-model capacity; a depth/colsample-diverse ensemble with many
# shallow-ish trees averages away the year-specific memorization and keeps improving with size.
CFG = [(3, 0.6), (4, 0.7), (5, 0.8), (6, 0.6), (3, 0.8), (4, 0.5),
       (5, 0.7), (6, 0.8), (3, 0.7), (4, 0.8), (5, 0.6), (6, 0.5)]

members = [
    xgb.XGBClassifier(
        n_estimators=1100,
        max_depth=d,
        learning_rate=0.05,
        colsample_bytree=c,
        max_bin=512,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    for i, (d, c) in enumerate(CFG)  # 12 members: one pass over the 12 configs
]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
for m in members:
    m.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
