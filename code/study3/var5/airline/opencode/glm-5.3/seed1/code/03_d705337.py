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
# base categoricals; hour_carr = interaction of departure hour with carrier
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "hour_carr"]
NUM_COLS = ["DepTime", "hour", "minute_of_day", "Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in
              ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
_hc = train["DepTime"].astype(int) // 100 % 24
_hc_levels = pd.Index(sorted((_hc.astype(str) + "_" + train["UniqueCarrier"].astype(str)).unique()))
cat_levels["hour_carr"] = _hc_levels


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dep = df["DepTime"].astype(int)
    hour = dep // 100 % 24
    X["hour_carr"] = pd.Categorical(hour.astype(str) + "_" + df["UniqueCarrier"].astype(str),
                                     categories=_hc_levels)
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute_of_day"] = dep // 100 * 60 + dep % 100
    X["Distance"] = df["Distance"].astype(float)
    return X[NUM_COLS + CAT_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: diverse 9-member ensemble (3 configs x 3 seeds) ------------------
X_tr, y_tr = prepare(train), to_y(train)
t0 = time.time()
members = []
for seed in (42, 1, 7):
    members += [
        xgb.XGBClassifier(
            n_estimators=900, max_depth=6, learning_rate=0.02, min_child_weight=20,
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
        ),
        xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, min_child_weight=20,
            reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, random_state=seed + 100, n_jobs=N_JOBS,
        ),
        xgb.XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.03, min_child_weight=40,
            reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True, random_state=seed + 200, n_jobs=N_JOBS,
        ),
    ]
models = [m.fit(X_tr, y_tr) for m in members]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
