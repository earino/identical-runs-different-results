"""XGBoost binary classifier for airline delays. Only file the agent edits.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]


# --- features -----------------------------------------------------------------
def _fe(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row feature engineering (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    dt = X["DepTime"]
    X["hour"] = (dt // 100) % 24
    X["minute"] = dt % 100
    X["Distance"] = df["Distance"].astype(float)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayofMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    return X


CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    X = _fe(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.7,
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
