"""Airline delay XGBoost. THIS IS THE ONLY FILE THE AGENT EDITS.

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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features (all stats fitted on train only) ---------------------------------
obj_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].unique())) for c in obj_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["day"] = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dep = df["DepTime"].astype(int)
    dm = ((dep // 100) * 60 + dep % 100) % 1440  # wrap anomalous 24:00+ times to early morning
    X["dep_minutes"] = dm
    X["hour"] = dm // 60
    X["minute"] = dm % 60
    X["distance"] = df["Distance"].astype(float)
    for c in obj_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.015,
    grow_policy="lossguide",
    max_leaves=8192,
    max_depth=0,
    colsample_bytree=0.55,
    max_bin=1024,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
