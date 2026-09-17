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
from sklearn.model_selection import KFold

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
# that transfer across years.
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
    X["DepMin"] = df["DepTime"].astype(int) % 100
    X["Distance"] = df["Distance"].astype(int)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, Xev = prepare(train), prepare(evald)
ytr = to_y(train)

# Diversity via depth / subsample / colsample, each with and without L1 (reg_alpha=1):
# averaging all ten full-data models beats any single model and CV-bagged variants.
MEMBERS = [
    dict(n_estimators=300, max_depth=6, learning_rate=0.10),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.7),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, colsample_bytree=0.8),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=300, max_depth=6, learning_rate=0.10, subsample=0.8),
    dict(n_estimators=300, max_depth=6, learning_rate=0.10, reg_alpha=1),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.7, reg_alpha=1),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, colsample_bytree=0.8, reg_alpha=1),
    dict(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1),
    dict(n_estimators=300, max_depth=6, learning_rate=0.10, subsample=0.8, reg_alpha=1),
]


def make_model(kw):
    return xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **kw,
    )


t0 = time.time()
models = []
for kw in MEMBERS:
    m = make_model(kw)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
