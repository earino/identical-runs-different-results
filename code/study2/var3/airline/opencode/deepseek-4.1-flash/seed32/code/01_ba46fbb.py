"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design: an ensemble of XGBoost models over native-categorical features plus smoothed target-encoding and
frequency features. All statistics (category levels, target-encoding, frequencies) are fit on the training
data at module level and then applied as fixed lookups inside prepare(), so predict_proba() reproduces the
exact same transformations on unseen data.
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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
_prior = float(y_train.mean())
_TE_COLS = ["Origin", "Dest", "UniqueCarrier"]
_TE_SMOOTH = 20.0


def _make_te_map(col: str) -> pd.Series:
    s = pd.DataFrame({"k": train[col].values, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    return (s["sum"] + _prior * _TE_SMOOTH) / (s["count"] + _TE_SMOOTH)


_te_maps = {c: _make_te_map(c) for c in _TE_COLS}
_freq_maps = {c: train[c].value_counts(normalize=True) for c in _TE_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in _TE_COLS:
        X["te_" + c] = df[c].map(_te_maps[c]).fillna(_prior)
        X["freq_" + c] = df[c].map(_freq_maps[c]).fillna(0.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Diverse low-learning-rate configurations; averaging reduces variance on the shifted holdout.
CONFIGS = [
    (160, 5, 0.02),
    (200, 6, 0.015),
    (300, 6, 0.01),
    (120, 6, 0.02),
    (80, 6, 0.03),
]

X_train = prepare(train)
t0 = time.time()
models = []
for n, depth, lr in CONFIGS:
    for seed in (1, 2):
        m = xgb.XGBClassifier(
            n_estimators=n,
            max_depth=depth,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=N_JOBS,
        )
        m.fit(X_train, y_train)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
