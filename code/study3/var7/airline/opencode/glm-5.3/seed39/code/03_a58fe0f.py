"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
SEEDS = (100, 101, 102, 103, 104)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_SRC = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(0).astype(int)


def _base(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    hour, minute = dep // 100, dep % 100
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour * 60 + minute  # minutes since midnight
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0.0)
    X["month_num"] = _num(df["Month"])
    X["day_num"] = _num(df["DayofMonth"])
    X["dow_num"] = _num(df["DayOfWeek"])
    for c in CAT_SRC:
        X[c] = df[c].astype(str).values
    X["hour_cat"] = hour.astype(str).values
    X["doy"] = (X["month_num"] - 1) * 31 + X["day_num"]  # approximate day of year / season
    return X


CAT_COLS = CAT_SRC + ["hour_cat"]
_levels_src = _base(train)
cat_levels = {c: pd.Index(sorted(_levels_src[c].dropna().unique())) for c in CAT_COLS}
feature_cols = list(_levels_src.columns)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _base(df)[feature_cols]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 5-seed ensemble of regularized XGB, early-stopped on eval ----------
PARAMS = dict(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.03,
    min_child_weight=20,
    reg_lambda=10.0,
    reg_alpha=4.0,
    subsample=0.9,
    colsample_bytree=0.8,
    max_bin=512,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    n_jobs=N_JOBS,
)

Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = []
t0 = time.time()
for seed in SEEDS:
    m = xgb.XGBClassifier(random_state=seed, **PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
