"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
N_SEEDS = 10

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; fitted stats come from module level (train only)."""
    X = pd.DataFrame(index=df.index)
    m = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    deptime = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = deptime // 100
    minute = deptime % 100
    dist = pd.to_numeric(df["Distance"], errors="coerce")

    X["Month"] = m
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["DepTime"] = deptime
    X["Hour"] = hour
    X["Minute"] = minute
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist)
    X["Month_sin"] = np.sin(2 * np.pi * m / 12)
    X["Dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["Dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["Hour_sin"] = np.sin(2 * np.pi * hour / 24)

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def features(df: pd.DataFrame) -> pd.DataFrame:
    return prepare(df)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed ensemble -----------------------------------------------------
params = dict(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
t0 = time.time()
Xfull = features(train)
yfull = to_y(train)
models = []
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(**{**params, "random_state": SEED + s})
    m.fit(Xfull, yfull)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_SEEDS} seeds)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = features(df)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
