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


def _hour_key(df: pd.DataFrame) -> pd.Series:
    hhmm = pd.to_numeric(df["DepTime"], errors="coerce")
    return df["Origin"].astype(str) + "_" + np.floor(hhmm / 100).astype("Int64").astype(str)


# Schedule density: how many departures an airport handles in that hour of the day. Hub volume in the evening
# bank is a stable, year-invariant congestion proxy.
_orig_hour_cnt = _hour_key(train).value_counts()


def _add_features(X: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    # hhmm departure -> hour / minute / minutes-since-midnight: hhmm is non-monotone (1259|1301) and hides
    # the strong "delays accumulate over the day" effect.
    hhmm = pd.to_numeric(src["DepTime"], errors="coerce")
    hour = np.floor(hhmm / 100)
    X["dep_hour"] = hour.clip(0, 24)
    X["dep_minute"] = hhmm - hour * 100
    X["dep_time_min"] = X["dep_hour"] * 60 + X["dep_minute"]
    # the calendar columns arrive as "c-<n>" levels; they are inherently ordered, so expose them as ordinals
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        X[c] = pd.to_numeric(src[c].astype(str).str.replace("c-", "", regex=False), errors="coerce").to_numpy()
    X["origin_hour_count"] = np.log1p(_hour_key(src).map(_orig_hour_cnt).fillna(0.0).to_numpy())
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X = _add_features(X, df)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# A bag of seed-diverse XGBoost models: averaging independent fits cancels part of the seed variance, which
# matters here because the target year differs from the training year.
SEEDS = (42, 7, 1234, 2024, 99, 5150, 17, 555, 808, 31337)


def _make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=20,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        early_stopping_rounds=50,
    )


Xtr, ytr = prepare(train), to_y(train)
Xev = prepare(evald)
t0 = time.time()
models = []
for seed in SEEDS:
    m = _make_model(seed)
    m.fit(Xtr, ytr, eval_set=[(Xev, to_y(evald))], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  best_iterations={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
