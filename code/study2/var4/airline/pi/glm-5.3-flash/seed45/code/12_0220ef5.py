"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # time-of-day features (strong signal: delay rate ramps from ~0.04 at 5am to ~0.8 late evening)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 29)
    minute = dep % 100
    mod = hour * 60 + minute  # minute of day
    X["DepTime_hour"] = hour
    X["DepTime_hour_cat"] = pd.Categorical(hour, categories=np.arange(31))
    X["DepTime_min"] = minute
    X["minute_of_day"] = mod
    ang = 2 * np.pi * mod / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    # numeric calendar
    X["Month_num"] = _num(df["Month"])
    X["DOW_num"] = _num(df["DayOfWeek"])
    X["DOM_num"] = _num(df["DayofMonth"])
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_dist"] = np.log1p(dist)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(n_estimators=800, max_depth=6, learning_rate=0.05),
    dict(n_estimators=800, max_depth=8, learning_rate=0.05),
    dict(n_estimators=400, max_depth=4, learning_rate=0.1),
    dict(n_estimators=800, max_depth=6, learning_rate=0.05, min_child_weight=10),
    dict(n_estimators=1200, max_depth=6, learning_rate=0.03),
    dict(n_estimators=600, max_depth=10, learning_rate=0.05),
    dict(n_estimators=300, max_depth=3, learning_rate=0.15),
    dict(n_estimators=1000, max_depth=8, learning_rate=0.03, min_child_weight=5),
    dict(n_estimators=200, max_depth=3, learning_rate=0.2),
    dict(n_estimators=1000, max_depth=12, learning_rate=0.03),
    dict(n_estimators=400, max_depth=6, learning_rate=0.1, min_child_weight=20),
    dict(n_estimators=800, max_depth=8, learning_rate=0.07),
]
models = []
t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
for i, cfg in enumerate(CONFIGS):
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  n_models={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
