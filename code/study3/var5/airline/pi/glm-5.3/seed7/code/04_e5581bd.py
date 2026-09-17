"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame (target col may be absent) -> 1-D P(positive) array.
     ALL feature engineering lives inside prepare(); encoders are fit on training data only.

Notes (from experiments 1-2 + local diagnostics):
  - 2005 -> 2006 is a real distribution shift: models that memorize 2005-specific route stats
    (route Origin_Dest categorical, route target encoding) lose ~0.01 AUC on 2006.
  - Early stopping on a 2005 slice rewards MORE rounds and hurts 2006 AUC; use fixed, moderate
    capacity instead: depth 4, lr 0.05, 200 rounds.
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

# --- encoders: fit on TRAIN ONLY -------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
MDAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
CUM = np.cumsum(np.concatenate([[0], MDAYS]))  # day-of-year via fixed non-leap calendar


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    t = df["DepTime"].astype(int)
    X["DepTime"] = t
    X["hour"] = t // 100
    X["dep_min"] = (t // 100) * 60 + t % 100
    X["Distance"] = df["Distance"].astype(float)
    mo = df["Month"].str.slice(2).astype(int).to_numpy()
    da = df["DayofMonth"].str.slice(2).astype(int).to_numpy()
    doy = CUM[mo - 1] + da
    X["doy"] = doy
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: fixed moderate capacity (no early stopping on a misleading 2005 slice).
# Small ensemble: colsample_bytree=0.7 gives the members real diversity; seeds alone do not.
N_MODELS = 5
t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
models = []
for s in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=s + 1,
        n_jobs=N_JOBS,
        min_child_weight=10,
        colsample_bytree=0.7,
    )
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
