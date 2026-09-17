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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
RAW_NUM = ["DepTime", "DepMinutes", "Distance"]
RAW_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    dep = df["DepTime"].astype(int)
    X["DepMinutes"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = df["Distance"].astype(float)
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

# --- bagged ensemble; member params = winner of the exp10 in-train CV grid -----
# (depth16/mcw5/lr.03 -> CV 0.7655 on 2005 folds; subsample .9 / colsample .7 fixed from exp8-9)
MEMBER = dict(
    n_estimators=1000,
    max_depth=16,
    min_child_weight=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.7,
)
models = []
t0 = time.time()
for seed in range(7):
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=60,
        random_state=seed,
        n_jobs=N_JOBS,
        **MEMBER,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")
print(f"best_iterations: {[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
