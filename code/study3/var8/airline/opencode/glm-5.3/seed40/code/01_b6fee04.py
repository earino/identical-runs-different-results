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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    # --- calendar features: c-<n> strings -> ints + cyclic sin/cos -------------------------
    for c, period in (("Month", 12), ("DayofMonth", 31), ("DayOfWeek", 7)):
        n = df[c].str.extract(r"c-(\d+)", expand=False).astype(float)
        X[c + "_n"] = n
        X[c + "_sin"] = np.sin(2 * np.pi * n / period)
        X[c + "_cos"] = np.cos(2 * np.pi * n / period)
    # --- departure time: hhmm int -> hour / minute / minutes-since-midnight + cyclic -------
    dt = df["DepTime"].astype(float)
    hh = (dt // 100).clip(0, 24)
    mm = dt - (dt // 100) * 100
    mins = (hh * 60 + mm).clip(0, 24 * 60)
    X["DepHour"] = hh
    X["DepMinute"] = mm
    X["DepMinutes"] = mins
    X["DepMinutes_sin"] = np.sin(2 * np.pi * mins / (24 * 60))
    X["DepMinutes_cos"] = np.cos(2 * np.pi * mins / (24 * 60))
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=30,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
n_val = 10000
va_idx = np.random.RandomState(SEED).choice(len(X_all), size=n_val, replace=False)
mask = np.ones(len(X_all), dtype=bool)
mask[va_idx] = False
model.fit(X_all[mask], y_all[mask], eval_set=[(X_all[~mask], y_all[~mask])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
