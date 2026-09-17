"""Airline delay XGBoost: engineered features + tuned model with early stopping.

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

# --- raw columns ----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["Distance"]


def _depsin(df: pd.DataFrame) -> np.ndarray:
    h = (df["DepTime"].fillna(0) // 100).to_numpy()
    m = (df["DepTime"].fillna(0) % 100).to_numpy()
    ang = 2 * np.pi * (h * 60 + m) / 1440.0
    return np.sin(ang)


def _depcos(df: pd.DataFrame) -> np.ndarray:
    h = (df["DepTime"].fillna(0) // 100).to_numpy()
    m = (df["DepTime"].fillna(0) % 100).to_numpy()
    ang = 2 * np.pi * (h * 60 + m) / 1440.0
    return np.cos(ang)


def _num_to_f32(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(np.float32)


train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
# category levels fitted on TRAIN only; unseen levels (eval/holdout) map to NaN
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; predict_proba() applies it to unseen rows."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    X["Distance"] = _num_to_f32(df, "Distance")
    X["DepTime"] = _num_to_f32(df, "DepTime")
    X["dep_sin"] = _depsin(df)
    X["dep_cos"] = _depcos(df)
    X["dep_hour"] = np.floor(df["DepTime"].fillna(0) / 100.0)
    X["dep_norm"] = (df["DepTime"].fillna(0) % 2400).to_numpy(np.float32)  # cyclic 2400
    X["log_dist"] = np.log1p(X["Distance"].to_numpy())
    X["dist_bin"] = pd.cut(
        pd.to_numeric(df["Distance"], errors="coerce").fillna(0),
        bins=[0, 150, 250, 400, 600, 900, 1300, 1900, 2600, 6000],
        labels=False,
    ).astype("float32")
    return X


FEATURE_COLS = [
    *CAT_COLS,
    "Distance",
    "DepTime",
    "dep_sin",
    "dep_cos",
    "dep_hour",
    "dep_norm",
    "log_dist",
    "dist_bin",
]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.7,
    gamma=0.3,
    reg_lambda=2.0,
    reg_alpha=0.1,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

# --- model ----------------------------------------------------------------------
y = to_y(train)
ye = to_y(evald)
Xtr = prepare(train)
Xev = prepare(evald)

t0 = time.time()
model.fit(
    Xtr,
    y,
    eval_set=[(Xev, ye)],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(ye, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
