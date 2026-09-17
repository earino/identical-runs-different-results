"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare(); encoders/statistics are fit on train only.
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
NUM_MONTHS = ["Month", "DayofMonth", "DayOfWeek"]


def _num(s: pd.Series) -> pd.Series:
    return s.str.slice(2).astype(int)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering: deterministic, no fitted state."""
    out = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(int)
    hour = (dt // 100).clip(0, 29)
    out["DepHour"] = hour
    out["DepMinutes"] = hour * 60 + dt % 100
    hh = out["DepMinutes"] % 1440
    out["DepSin"] = np.sin(2 * np.pi * hh / 1440)
    out["DepCos"] = np.cos(2 * np.pi * hh / 1440)
    out["DistanceLog"] = np.log1p(df["Distance"])
    return out


# fitted state: category levels, learned from TRAIN ONLY
CAT_LEVELS = {}
for c in CAT_COLS:
    if c == "Route":
        vals = train["Origin"] + "_" + train["Dest"]
    else:
        vals = train[c]
    CAT_LEVELS[c] = pd.Index(sorted(vals.astype(str).unique()))
for c in NUM_MONTHS:
    CAT_LEVELS[c] = pd.Index(sorted(train[c].astype(str).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = engineer(df)
    for c in CAT_COLS + NUM_MONTHS:
        if c == "Route":
            vals = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
        else:
            vals = df[c].astype(str)
        X[c] = pd.Categorical(vals, categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
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
