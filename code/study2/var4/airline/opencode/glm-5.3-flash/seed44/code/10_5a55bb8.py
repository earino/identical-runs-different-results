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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
cat_levels["ch"] = pd.Index(sorted((train["UniqueCarrier"].astype(object) + "_" + (train["DepTime"] // 100).astype(str)).unique()))
_h30 = (train["DepTime"] // 25).astype(str)
cat_levels["ch30"] = pd.Index(sorted((train["UniqueCarrier"].astype(object) + "_" + _h30).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["Distance"])
    hour = df["DepTime"].astype(float) // 100
    minute = df["DepTime"].astype(float) % 100
    X["hour"] = hour
    X["minute"] = minute
    mod = hour * 60 + minute
    X["minofday"] = mod
    X["tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    month = df["Month"].str.slice(2).astype(float)
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    dow = df["DayOfWeek"].str.slice(2).astype(float)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["day"] = df["DayofMonth"].str.slice(2).astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["ch"] = pd.Categorical(df["UniqueCarrier"].astype(object) + "_" + (df["DepTime"].astype(float) // 100).astype(int).astype(str), categories=cat_levels["ch"])
    X["ch30"] = pd.Categorical(df["UniqueCarrier"].astype(object) + "_" + (df["DepTime"].astype(float) // 25).astype(int).astype(str), categories=cat_levels["ch30"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = len(train) // 10
val_idx, fit_idx = idx[:n_val], idx[n_val:]
Xf, yf = prepare(train), to_y(train)

model = xgb.XGBClassifier(
    n_estimators=4000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xf, yf, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
