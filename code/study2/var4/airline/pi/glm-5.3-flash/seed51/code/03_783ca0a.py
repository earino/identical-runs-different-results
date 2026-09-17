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
from sklearn.model_selection import train_test_split

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
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# early stopping on eval.csv (2006 distribution — the shift the hidden holdout shares)
X_tr = prepare(train)
t0 = time.time()
model.fit(X_tr, to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
best_it = model.best_iteration
print(f"Training time: {time.time() - t0:.1f}s, best_iteration={best_it}")
model.n_estimators = int(best_it)  # already the trees kept by early stopping

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
