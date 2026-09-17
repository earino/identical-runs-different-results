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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]
cat_levels = {}


def _add_route(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    return df


train = _add_route(train)
evald = _add_route(evald)
for c in CAT_COLS:
    cat_levels[c] = pd.Index(sorted(train[c].dropna().unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    df = _add_route(df)
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = df["DepTime"].fillna(-1)
    hour = (dt // 100).clip(-1, 47)
    minute = dt % 100
    lin = hour * 60 + minute
    X["dep_sin"] = np.sin(2 * np.pi * lin / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * lin / 1440.0)
    X["hour_cat"] = pd.Categorical(hour.clip(0, 47).astype(int).astype(str).where(dt >= 0, None))
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    dom = df["DayofMonth"].str.replace("c-", "", regex=False).astype(float)
    X["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    X["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
    mon = df["Month"].str.replace("c-", "", regex=False).astype(float)
    X["mon_sin"] = np.sin(2 * np.pi * (mon - 1) / 12.0)
    X["mon_cos"] = np.cos(2 * np.pi * (mon - 1) / 12.0)
    dow = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(float)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train),
    to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s (best iter {model.best_iteration})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
p = predict_proba(evald)
eval_auc = roc_auc_score(to_y(evald), p)
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC5: {eval_auc:.5f}")
print(f"Eval AUC: {eval_auc:.4f}")
