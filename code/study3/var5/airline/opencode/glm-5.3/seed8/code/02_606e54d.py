"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Key findings baked in:
  - Month and DayofMonth are DROPPED: their 2005 seasonal delay patterns do not transfer to 2006.
  - One-hot (not native categorical) for DayOfWeek/UniqueCarrier/Origin/Dest.
  - Seed-bagged ensemble of 5 XGB models (variance reduction across the 2005->2006 shift).
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
NUM_COLS = ["DepTime", "Distance"]
CAT_COLS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]  # Month/DayofMonth dropped on purpose
levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}  # fit on train only


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    parts = [df[NUM_COLS].astype(float)]
    for c in CAT_COLS:
        parts.append(pd.get_dummies(pd.Categorical(df[c], categories=levels[c]), prefix=c))
    return pd.concat(parts, axis=1)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed-bagged ensemble ----------------------------------------------
BASE = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    tree_method="hist",
)
N_MODELS = 5

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
models = []
for s in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=s, n_jobs=N_JOBS, **BASE)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
