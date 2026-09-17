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
# DayofMonth delay rates do not transfer across years (corr 0.33) -> drop it as noise.
DROP = ["DayofMonth", "Month"]
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET] + DROP]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: heterogeneous bagged ensemble of shallow XGBoost trees -------------
# Members vary depth / subsampling so their errors decorrelate (2005->2006 shift).
MEMBERS = [
    dict(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=3, n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7),
    dict(max_depth=6, n_estimators=300, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7),
    dict(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.7, colsample_bytree=1.0),
    dict(max_depth=3, n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6),
    dict(max_depth=5, n_estimators=300, learning_rate=0.04, subsample=0.7, colsample_bytree=0.6),
    dict(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6),
]
SEEDS = [42, 7, 13, 99, 2024, 17, 23, 31]


def _make_model(seed: int, **kw) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        **kw,
    )


t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for _s, _kw in zip(SEEDS, MEMBERS):
    _m = _make_model(_s, **_kw)
    _m.fit(X_train, y_train)
    models.append(_m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
