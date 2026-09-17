"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

FEATURE_COLS = [
    "DepHour", "DepMinute", "DepMinOfDay", "DepSinH", "DepCosH",
    "DepSinD", "DepCosD",
    "Month", "DayOfMonth", "DayOfWeek",
    "Distance", "LogDistance",
    "UniqueCarrier", "Origin", "Dest",
]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {}  # fitted on training data only; unseen levels map to NaN


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = np.floor(dt / 100.0)
    mm = dt - hh * 100
    # scheduled times >= 2400 (after-midnight roll) wrap into the early morning
    hh = np.where(hh >= 24, hh - 24, hh)
    hour = hh
    mod = hour * 60 + mm  # minutes of day, cyclic in [0, 1440)
    X["DepHour"] = hour
    X["DepMinute"] = mm
    X["DepMinOfDay"] = mod
    X["DepSinH"] = np.sin(2 * np.pi * hour / 24.0)
    X["DepCosH"] = np.cos(2 * np.pi * hour / 24.0)
    X["DepSinD"] = np.sin(2 * np.pi * mod / 1440.0)
    X["DepCosD"] = np.cos(2 * np.pi * mod / 1440.0)
    for c, name in (("Month", "Month"), ("DayofMonth", "DayOfMonth"), ("DayOfWeek", "DayOfWeek")):
        X[name] = pd.to_numeric(df[c].astype(str).str.lstrip("c-"), errors="coerce")
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist)
    for c in CAT_COLS:
        levels = CAT_LEVELS.get(c)
        if levels is None:  # first call is on the training frame: fit
            levels = pd.Index(sorted(df[c].dropna().astype(str).unique()))
            CAT_LEVELS[c] = levels
        X[c] = pd.Categorical(df[c].astype("object").where(df[c].notna(), None).astype(str),
                              categories=levels)
    return X[FEATURE_COLS]


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

t0 = time.time()
# early stopping on a chronological split: last 20% of the 2005 training rows
# mimics the 2005 -> 2006 shift better than a random split
tr = train.iloc[:80000]
va = train.iloc[80000:]
model.fit(prepare(tr), to_y(tr), eval_set=[(prepare(va), to_y(va))], verbose=False)
print(f"Best iteration: {model.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
