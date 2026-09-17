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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]


# --- features -----------------------------------------------------------------
def _fe(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row feature engineering (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayofMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    dt = df["DepTime"].astype(int)
    hour = (dt // 100) % 24
    minute = dt % 100
    X["DepTime"] = dt
    X["hour"] = hour
    X["minute"] = minute
    ang = 2 * np.pi * (hour + minute / 60.0) / 24.0
    X["hour_sin"] = np.sin(ang)
    X["hour_cos"] = np.cos(ang)
    X["is_night"] = ((hour >= 22) | (hour <= 4)).astype(int)
    X["is_weekend"] = (X["DayOfWeek"] >= 6).astype(int)
    X["Route"] = df["Origin"] + "_" + df["Dest"]
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    X["dist_per_min"] = df["Distance"].astype(float) / (1.0 + dt)
    for c in ["UniqueCarrier", "Origin", "Dest", "Route", "Month", "DayofMonth", "DayOfWeek"]:
        X[c] = X[c].astype(str) if c == "Route" else df[c]
    return X


CAT_LEVELS = {
    c: pd.Index(sorted(train[c].astype(str).unique()))
    for c in CAT_COLS
    if c != "Route"
}
CAT_LEVELS["Route"] = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    X = _fe(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=CAT_LEVELS[c].astype(str))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=20,
    subsample=0.9,
    colsample_bytree=0.9,
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
