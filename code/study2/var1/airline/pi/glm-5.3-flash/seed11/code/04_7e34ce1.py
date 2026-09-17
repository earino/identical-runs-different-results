"""XGBoost airline-delay classifier, v2: engineered features + bigger model + early stopping.

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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# Categories are fit on TRAINING data only; unseen levels map to NaN (handled by XGBoost).
CAT_LEVELS = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().astype(str).unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().astype(str).unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().astype(str).unique())),
}


def _num(s):
    # 'c-7' -> 7
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    X["Month"] = month
    X["DayOfMonth"] = dom
    X["DayOfWeek"] = dow
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 24)
    minute = dt % 100
    X["DepHour"] = hour + minute / 60.0
    ang = 2.0 * np.pi * X["DepHour"].fillna(12.0) / 24.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    X["RedEye"] = ((hour >= 0) & (hour <= 5)).astype(float)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=CAT_LEVELS["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"].astype(str), categories=CAT_LEVELS["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"].astype(str), categories=CAT_LEVELS["Dest"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=20,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s (trees={model.best_iteration + 1})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
