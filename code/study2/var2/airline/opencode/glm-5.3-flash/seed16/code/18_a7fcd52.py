"""XGBoost binary classifier for airline delay. Only file the agent edits.

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
# categorical source columns and the interaction cats we build from them
CAT_SRC = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
COMBOS = [
    ("Origin", "Dest"),
    ("UniqueCarrier", "Hour"),
    ("Origin", "Hour"),
    ("Dest", "Hour"),
]
CAT_LEVELS = {}


def _base_cols(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype("int64")
    hour = (dt // 100) % 24
    minute = dt % 100
    t = hour * 60 + minute
    X["DepTime_sin"] = np.sin(2 * np.pi * t / 1440)
    X["DepTime_cos"] = np.cos(2 * np.pi * t / 1440)
    X["Hour"] = hour
    X["Minute"] = minute
    X["Distance"] = np.log1p(df["Distance"].astype("float64"))
    for c in CAT_SRC:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def build_levels() -> None:
    for c in CAT_SRC:
        CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().unique()))
    b = _base_cols(train)
    for c in COMBOS:
        name = "+".join(c)
        CAT_LEVELS[name] = pd.Index(
            (b[c[0]].astype(str) + "_" + b[c[1]].astype(str)).drop_duplicates().sort_values()
        )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _base_cols(df)
    for c in CAT_SRC:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    for c in COMBOS:
        name = "+".join(c)
        X[name] = pd.Categorical(
            X[c[0]].astype(str) + "_" + X[c[1]].astype(str), categories=CAT_LEVELS[name]
        )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


build_levels()

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=4400,
    learning_rate=0.05,
    max_depth=7,
    tree_method="hist",
    max_bin=64,
    colsample_bynode=0.3,
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
