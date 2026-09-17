"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# key insight: interaction of fine time-of-day bins (15/30-minute) with carrier is the strongest signal
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "b15_carr", "b30_carr"]
NUM_COLS = ["DepTime", "hour", "minute_of_day", "Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in
              ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
_dep = train["DepTime"].astype(int)
_carr = train["UniqueCarrier"].astype(str)


def _binkey(dep: pd.Series, minutes: int) -> pd.Series:
    return dep // 100 * (60 // minutes) + (dep % 100) // minutes


for _m in (15, 30):
    cat_levels[f"b{_m}_carr"] = pd.Index(sorted((_binkey(_dep, _m).astype(str) + "_" + _carr).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dep = df["DepTime"].astype(int)
    carr = df["UniqueCarrier"].astype(str)
    for m in (15, 30):
        X[f"b{m}_carr"] = pd.Categorical(_binkey(dep, m).astype(str) + "_" + carr,
                                         categories=cat_levels[f"b{m}_carr"])
    X["DepTime"] = dep
    X["hour"] = dep // 100 % 24
    X["minute_of_day"] = dep // 100 * 60 + dep % 100
    X["Distance"] = df["Distance"].astype(float)
    return X[NUM_COLS + CAT_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 2 configs x 3 seeds ensemble ---------------------------------------
BASE = dict(min_child_weight=20, reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True)
CONFIGS = [
    dict(n_estimators=900, max_depth=6, learning_rate=0.02),
    dict(n_estimators=300, max_depth=8, learning_rate=0.05),
]
X_tr, y_tr = prepare(train), to_y(train)
t0 = time.time()
models = []
for seed in (42, 1, 7):
    for kw in CONFIGS:
        m = xgb.XGBClassifier(random_state=seed, n_jobs=N_JOBS, **BASE, **kw)
        m.fit(X_tr, y_tr)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
