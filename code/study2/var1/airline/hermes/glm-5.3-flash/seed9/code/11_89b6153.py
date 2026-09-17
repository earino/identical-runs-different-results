"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[CAT_COLS + ["DepTime", "Distance"]].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# heterogeneous bagging: jitter subsample/colsample/depth per member for less-correlated trees
_MIX = [
    (0.85, 0.8, 6),
    (0.70, 0.6, 6),
    (0.90, 0.9, 5),
    (0.80, 0.7, 7),
    (0.75, 0.9, 6),
    (0.85, 0.6, 6),
    (0.70, 0.8, 5),
    (0.90, 0.7, 7),
    (0.80, 0.8, 5),
    (0.75, 0.6, 7),
    (0.85, 0.9, 7),
    (0.70, 0.7, 6),
]

t0 = time.time()
models = []
for i, (ss, cs, md) in enumerate(_MIX):
    m = xgb.XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=md,
        subsample=ss,
        colsample_bytree=cs,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=42 + i,
        n_jobs=N_JOBS,
    )
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
