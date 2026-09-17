"""XGBoost binary classifier for airline departure delay prediction.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: bagged ensemble of shallow-ish XGBoost models (row/col subsampling, seed diversity,
config diversity). Month and DayofMonth are dropped: their delay-rate effects do not
transfer from 2005 (train) to 2006 (eval/holdout).
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
# Month and DayofMonth are deliberately NOT used (unstable year-over-year; see experiments log).
CAT_COLS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble ------------------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
CONFIGS = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8),
    dict(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8),
    dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.7),
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=1.0, colsample_bytree=0.9),
]
N_SEEDS = 12

X_all, y_all = prepare(train), to_y(train)
X_eval = prepare(evald)

models = []
t0 = time.time()
for cfg in CONFIGS:
    for seed in range(N_SEEDS):
        m = xgb.XGBClassifier(random_state=seed, **cfg, **BASE)
        m.fit(X_all, y_all)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
