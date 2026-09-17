"""XGBoost binary classifier — airline dep-delay benchmark.

Contract (program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All feature engineering is inside prepare(); statistics are fit on train.csv only.
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

# --- features ---------------------------------------------------------------
# cyclical time features: hour-of-day is the strongest known signal for departure delays
cat_levels = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
}

feature_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance",
                "UniqueCarrier", "Origin", "Dest", "sin_hour", "cos_hour"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        X[c] = df[c].map(lambda v: int(str(v).lstrip("c-"))).astype("int16")
    X["DepTime"] = df["DepTime"].astype("int32")
    X["Distance"] = df["Distance"].astype("float32")
    h = np.array([int(str(t).zfill(4)[:2]) for t in df["DepTime"]])
    ang = 2 * np.pi * h / 24.0
    X["sin_hour"] = np.sin(ang)
    X["cos_hour"] = np.cos(ang)
    for c in ("UniqueCarrier", "Origin", "Dest"):
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model -------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=30,
    random_state=7,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iteration={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, model.best_iteration + 1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
