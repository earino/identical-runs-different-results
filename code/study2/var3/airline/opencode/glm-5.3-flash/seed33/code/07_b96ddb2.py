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
# CarQ = carrier x departure-hour x quarter-hour: carrier-specific schedule patterns
# that transfer across years. CarHour subsumed by CarQ, also dropped.
CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepHour", "CarQ"]


def build_levels() -> dict:
    lv = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS if c not in ("DepHour", "CarQ")}
    h = ((train["DepTime"] // 100).clip(0, 29)).astype(str)
    q = (train["DepTime"] % 100) // 15
    lv["DepHour"] = pd.Index(sorted(h.unique()))
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


Xtr, Xev = prepare(train), prepare(evald)


def fit(n_estimators, learning_rate, max_depth=5, subsample=1.0, colsample_bytree=1.0):
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


t0 = time.time()
# diversity via depth / subsample / colsample: averaging these five beats any single model
models = [
    fit(300, 0.10, max_depth=6),
    fit(600, 0.05, subsample=0.7),
    fit(600, 0.05, colsample_bytree=0.8),
    fit(600, 0.05, subsample=0.8, colsample_bytree=0.8),
    fit(300, 0.10, max_depth=6, subsample=0.8),
]
for m in models:
    m.fit(Xtr, to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    p = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
