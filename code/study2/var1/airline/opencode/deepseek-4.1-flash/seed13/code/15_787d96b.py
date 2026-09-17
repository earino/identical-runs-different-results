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
_dm_train = (train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)
_ch_train = train["UniqueCarrier"].astype(str) + "_" + (_dm_train // 30).astype(str)
ch_levels = pd.Index(sorted(_ch_train.unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dep = df["DepTime"].to_numpy()
    X["dep_min"] = (dep // 100) * 60 + (dep % 100)
    X["hour"] = dep // 100
    X["dep_minute"] = dep % 100
    dm = (dep // 100) * 60 + (dep % 100)
    ch = df["UniqueCarrier"].astype(str) + "_" + (dm // 30).astype(str)
    X["carrier_hour"] = pd.Categorical(ch, categories=ch_levels)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_train = prepare(train)
y_train = to_y(train)
X_eval = prepare(evald)
y_eval = to_y(evald)


CONFIGS = [
    dict(seed=42, max_depth=4, subsample=0.8, colsample_bytree=0.5),
    dict(seed=7, max_depth=5, subsample=0.8, colsample_bytree=0.5),
    dict(seed=2024, max_depth=5, subsample=0.7, colsample_bytree=0.5),
    dict(seed=123, max_depth=6, subsample=0.8, colsample_bytree=0.5),
    dict(seed=999, max_depth=6, subsample=0.7, colsample_bytree=0.5),
    dict(seed=555, max_depth=7, subsample=0.7, colsample_bytree=0.5),
    dict(seed=321, max_depth=7, subsample=0.9, colsample_bytree=0.5),
]


def fit_one(cfg: dict) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        min_child_weight=5,
        reg_lambda=5.0,
        gamma=0.0,
        max_cat_threshold=256,
        max_cat_to_onehot=32,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
        max_depth=cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
    )
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    return m


t0 = time.time()
models = [fit_one(c) for c in CONFIGS]
print(f"Training time: {time.time() - t0:.1f}s  n_models={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean(
        [m.predict_proba(X, iteration_range=(0, m.best_iteration + 1))[:, 1] for m in models],
        axis=0,
    )


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
