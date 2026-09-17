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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _c_int(s: pd.Series) -> pd.Series:
    return s.str.slice(2).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _c_int(df["Month"])
    dom = _c_int(df["DayofMonth"])
    dow = _c_int(df["DayOfWeek"])
    dt = df["DepTime"].astype(float)
    hour = (dt // 100) % 24
    minute = dt % 100
    tfrac = (hour + minute / 60.0) / 24.0
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["hour"] = hour
    X["minute"] = minute
    X["t_sin"] = np.sin(2 * np.pi * tfrac)
    X["t_cos"] = np.cos(2 * np.pi * tfrac)
    X["distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["distance"])
    X["dist_bin"] = pd.cut(X["distance"], bins=[0, 250, 500, 750, 1000, 1500, 2500, 1e9], labels=False)
    X["red_eye"] = ((hour < 6) | (dt >= 2400)).astype(int)
    X["evening"] = (hour >= 19).astype(int)
    X["min_on_qtr"] = (minute % 15 == 0).astype(int)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)

BASE = dict(
    n_estimators=20000,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=20,
    reg_lambda=2.0,
    gamma=1.0,
    tree_method="hist",
    eval_metric="auc",
    enable_categorical=True,
    early_stopping_rounds=400,
    n_jobs=N_JOBS,
)

MODELS = [
    dict(max_depth=10, learning_rate=0.02, random_state=42),
    dict(max_depth=12, learning_rate=0.015, random_state=7),
    dict(max_depth=14, learning_rate=0.01, random_state=123),
]

models = []
t0 = time.time()
for cfg in MODELS:
    m = xgb.XGBClassifier(**{**BASE, **cfg})
    m.fit(X_all, y_all, eval_set=[(X_ev, y_ev)], verbose=False)
    models.append(m)
    print(f"trained depth={cfg['max_depth']} best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
