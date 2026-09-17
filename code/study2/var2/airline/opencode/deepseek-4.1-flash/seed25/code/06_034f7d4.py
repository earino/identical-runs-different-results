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


def _freq_map(keys: pd.Series) -> pd.Series:
    return keys.astype(str).value_counts()


_freq_origin = _freq_map(train["Origin"])
_freq_dest = _freq_map(train["Dest"])
_freq_carrier = _freq_map(train["UniqueCarrier"])
_freq_route = _freq_map(train["Origin"].astype(str) + "_" + train["Dest"].astype(str))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    t = df["DepTime"].astype(int).clip(0, 2400)
    X["dep_hour"] = (t // 100).clip(0, 23).astype(int)
    X["dep_min"] = (t % 100).clip(0, 59).astype(int)
    X["dep_tod"] = X["dep_hour"] * 60 + X["dep_min"]
    X["freq_origin"] = df["Origin"].astype(str).map(_freq_origin).fillna(0).astype(float)
    X["freq_dest"] = df["Dest"].astype(str).map(_freq_dest).fillna(0).astype(float)
    X["freq_carrier"] = df["UniqueCarrier"].astype(str).map(_freq_carrier).fillna(0).astype(float)
    X["freq_route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(_freq_route).fillna(0).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=4, learning_rate=0.03, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, random_state=42),
    dict(max_depth=4, learning_rate=0.03, min_child_weight=10, subsample=0.8, colsample_bytree=0.6, random_state=7),
    dict(max_depth=5, learning_rate=0.03, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, random_state=13),
    dict(max_depth=3, learning_rate=0.03, min_child_weight=5, subsample=0.9, colsample_bytree=0.8, random_state=21),
    dict(max_depth=6, learning_rate=0.03, min_child_weight=20, subsample=0.8, colsample_bytree=0.7, random_state=34),
    dict(max_depth=4, learning_rate=0.05, min_child_weight=15, subsample=0.7, colsample_bytree=0.9, random_state=55),
    dict(max_depth=5, learning_rate=0.02, min_child_weight=8, subsample=0.9, colsample_bytree=0.6, random_state=89),
    dict(max_depth=4, learning_rate=0.04, min_child_weight=10, subsample=0.6, colsample_bytree=0.8, random_state=144),
]

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = []
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=5000,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"  depth={cfg['max_depth']} seed={cfg['random_state']} best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
