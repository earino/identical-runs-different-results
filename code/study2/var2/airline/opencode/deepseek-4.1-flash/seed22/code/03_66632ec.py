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

# --- features -----------------------------------------------------------------
# Month/DayofMonth are deliberately excluded: their 2005 seasonality does not transfer to the
# time-separated evaluation data and they hurt AUC. DayOfWeek does transfer.
NUM_COLS = ["DepTime", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "DayOfWeek"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])  # unseen levels -> NaN
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (t // 100).clip(0, 23)
    minute = (t % 100).clip(0, 59)
    tod = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["dep_time_min"] = tod
    X["dep_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
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
