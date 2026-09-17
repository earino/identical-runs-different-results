"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _clock(df: pd.DataFrame) -> tuple:
    """hhmm scheduled departure -> (hour, minute, time-of-day float)."""
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    t = t.where(t < 2400, t - 2400)  # 24xx (after midnight) -> 00xx
    hour = (t // 100).clip(0, 23)
    minute = (t % 100).clip(0, 59)
    return hour, minute, hour + minute / 60.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = df[CAT_COLS + ["Distance"]].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    hour, minute, tod = _clock(df)
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.02,
    colsample_bytree=0.3,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

model.fit(prepare(train), to_y(train))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


if __name__ == "__main__":
    evald = pd.read_csv("data/eval.csv")
    eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
    print(f"Eval AUC: {eval_auc:.4f}")
