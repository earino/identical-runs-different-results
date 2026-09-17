"""XGBoost binary classifier with engineered features. Only file the agent edits.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {}


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied identically everywhere (train/eval/holdout)."""
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(float) % 2400
    X["deptime"] = dt
    X["hour"] = (dt // 100).astype(int)
    X["minute"] = (dt % 100).astype(int)
    ang = 2 * np.pi * (dt / 1440.0)
    X["dt_sin"] = np.sin(ang)
    X["dt_cos"] = np.cos(ang)
    X["distance"] = df["Distance"].astype(float)
    X["logdist"] = np.log1p(X["distance"])
    for c in CAT_COLS:
        X[c] = df[c]
    return X


def fit_state() -> None:
    """Fit encoders on TRAINING data only."""
    for c in CAT_COLS:
        CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows."""
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


fit_state()

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
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
