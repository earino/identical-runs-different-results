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

# Month is DROPPED: month-of-year delay patterns do not transfer from 2005 to 2006.
# CarHour = carrier x departure-hour; CarQ = carrier x hour x quarter-hour:
# carrier-specific schedule delay patterns (transfer across years).
CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepHour", "CarHour", "CarQ"]


def build_levels() -> dict:
    lv = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS if c not in ("DepHour", "CarHour", "CarQ")}
    h = ((train["DepTime"] // 100).clip(0, 29)).astype(str)
    q = (train["DepTime"] % 100) // 15
    lv["DepHour"] = pd.Index(sorted(h.unique()))
    lv["CarHour"] = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + h).unique()))
    lv["CarQ"] = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + h + "_" + q.astype(str)).unique()))
    return lv


CAT_LEVELS = build_levels()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    h = ((df["DepTime"] // 100).clip(0, 29)).astype(str)
    q = (df["DepTime"] % 100) // 15
    for c in CAT_COLS:
        if c == "DepHour":
            vals = h
        elif c == "CarHour":
            vals = df["UniqueCarrier"].astype(str) + "_" + h
        elif c == "CarQ":
            vals = df["UniqueCarrier"].astype(str) + "_" + h + "_" + q.astype(str)
        else:
            vals = df[c].astype(str)
        X[c] = pd.Categorical(vals, categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(int)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
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
