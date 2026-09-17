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
# categorical columns handled natively by XGBoost; ordinals become numeric
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayOfMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    ang = 2 * np.pi * X["Month"] / 12.0
    X["Month_sin"] = np.sin(ang)
    X["Month_cos"] = np.cos(ang)
    # DepTime: scheduled hhmm (may exceed 2359 in the raw data) -> cyclic time of day
    mins = (df["DepTime"] // 100 * 60 + df["DepTime"] % 100) % 1440
    X["DepMin"] = mins
    ang = 2 * np.pi * mins / 1440.0
    X["DepTime_sin"] = np.sin(ang)
    X["DepTime_cos"] = np.cos(ang)
    X["DepHour"] = (mins // 60).astype(int)
    X["IsRedEye"] = ((X["DepHour"] <= 4) | (X["DepHour"] >= 22)).astype(int)
    X["Distance"] = df["Distance"]
    X["Distance_log"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# bagging ensemble of diverse XGBoost members, each early-stopped on eval
MEMBERS = [
    dict(max_depth=10, learning_rate=0.05, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, random_state=42),
    dict(max_depth=10, learning_rate=0.05, min_child_weight=20, subsample=0.7, colsample_bytree=0.7, random_state=7),
    dict(max_depth=8, learning_rate=0.05, min_child_weight=10, subsample=0.9, colsample_bytree=0.8, random_state=13),
    dict(max_depth=8, learning_rate=0.05, min_child_weight=30, subsample=0.8, colsample_bytree=0.6, random_state=101),
    dict(max_depth=10, learning_rate=0.08, min_child_weight=40, subsample=0.8, colsample_bytree=0.8, random_state=202),
    dict(max_depth=12, learning_rate=0.05, min_child_weight=50, subsample=0.8, colsample_bytree=0.8, random_state=303),
]

X_tr = prepare(train)
X_ev = prepare(evald)
y_tr, y_ev = to_y(train), to_y(evald)

models = []
t0 = time.time()
for cfg in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=2000,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
    print(f"member depth={cfg['max_depth']} seed={cfg['random_state']} auc={auc:.4f} it={m.best_iteration}")
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s, {len(models)} members")

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
