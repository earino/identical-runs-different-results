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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if c in ("UniqueCarrier", "Origin", "Dest")]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _cal(df: pd.DataFrame, c: str) -> pd.Series:
    return df[c].astype(str).str.split("-").str[1].astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    month = _cal(df, "Month")
    dom = _cal(df, "DayofMonth")
    dow = _cal(df, "DayOfWeek")
    doy = (month - 1) * 31 + dom
    X["month_n"] = month
    X["dom_n"] = dom
    X["dow_n"] = dow
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
SPECS = [
    dict(n_estimators=30, max_depth=3, learning_rate=0.1, subsample=1.0, colsample=1.0, seed=1),
    dict(n_estimators=30, max_depth=4, learning_rate=0.1, subsample=1.0, colsample=1.0, seed=2),
    dict(n_estimators=30, max_depth=5, learning_rate=0.1, subsample=1.0, colsample=1.0, seed=3),
    dict(n_estimators=30, max_depth=6, learning_rate=0.1, subsample=1.0, colsample=1.0, seed=4),
    dict(n_estimators=60, max_depth=4, learning_rate=0.05, subsample=0.8, colsample=0.8, seed=5),
    dict(n_estimators=60, max_depth=6, learning_rate=0.05, subsample=0.8, colsample=0.8, seed=6),
    dict(n_estimators=15, max_depth=4, learning_rate=0.2, subsample=0.8, colsample=0.8, seed=7),
    dict(n_estimators=100, max_depth=3, learning_rate=0.03, subsample=0.8, colsample=0.8, seed=8),
    dict(n_estimators=80, max_depth=5, learning_rate=0.05, subsample=0.8, colsample=0.8, seed=9),
    dict(n_estimators=80, max_depth=6, learning_rate=0.05, subsample=0.8, colsample=0.8, seed=10),
    dict(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample=0.8, seed=11),
    dict(n_estimators=30, max_depth=2, learning_rate=0.1, subsample=1.0, colsample=1.0, seed=12),
    dict(n_estimators=120, max_depth=6, learning_rate=0.04, subsample=0.7, colsample=0.7, mcw=10, lam=3.0, seed=13),
    dict(n_estimators=100, max_depth=7, learning_rate=0.04, subsample=0.7, colsample=0.7, mcw=10, lam=3.0, seed=14),
    dict(n_estimators=80, max_depth=8, learning_rate=0.04, subsample=0.7, colsample=0.7, mcw=20, lam=5.0, seed=15),
    dict(n_estimators=60, max_depth=5, learning_rate=0.06, subsample=0.7, colsample=0.7, mcw=20, lam=5.0, seed=16),
]
X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for sp in SPECS:
    m = xgb.XGBClassifier(
        n_estimators=sp["n_estimators"],
        max_depth=sp["max_depth"],
        learning_rate=sp["learning_rate"],
        subsample=sp["subsample"],
        colsample_bytree=sp["colsample"],
        min_child_weight=sp.get("mcw", 1),
        reg_lambda=sp.get("lam", 1.0),
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + sp["seed"],
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
