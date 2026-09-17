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
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def make_model(spec):
    return xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **spec,
    )


# --- search on an internal 70/30 split (no eval peeking) -----------------------
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
cut = int(len(train) * 0.7)
tr_i, va_i = idx[:cut], idx[cut:]
Xtr, ytr = prepare(train.iloc[tr_i]), to_y(train.iloc[tr_i])
Xv, yv = prepare(train.iloc[va_i]), to_y(train.iloc[va_i])

grid = [
    dict(n_estimators=400, max_depth=9, learning_rate=0.02, subsample=0.75, colsample_bytree=0.55, min_child_weight=40, reg_lambda=5),
    dict(n_estimators=600, max_depth=9, learning_rate=0.02, subsample=0.75, colsample_bytree=0.55, min_child_weight=40, reg_lambda=5),
    dict(n_estimators=600, max_depth=9, learning_rate=0.015, subsample=0.75, colsample_bytree=0.55, min_child_weight=40, reg_lambda=5),
    dict(n_estimators=800, max_depth=9, learning_rate=0.015, subsample=0.75, colsample_bytree=0.55, min_child_weight=40, reg_lambda=5),
    dict(n_estimators=600, max_depth=10, learning_rate=0.02, subsample=0.75, colsample_bytree=0.5, min_child_weight=60, reg_lambda=5),
    dict(n_estimators=400, max_depth=9, learning_rate=0.02, subsample=0.8, colsample_bytree=0.6, min_child_weight=20, reg_lambda=5),
    dict(n_estimators=600, max_depth=9, learning_rate=0.02, subsample=0.7, colsample_bytree=0.55, min_child_weight=40, reg_lambda=3),
    dict(n_estimators=600, max_depth=11, learning_rate=0.02, subsample=0.75, colsample_bytree=0.55, min_child_weight=80, reg_lambda=5),
    dict(n_estimators=600, max_depth=9, learning_rate=0.02, subsample=0.75, colsample_bytree=0.55, min_child_weight=40, reg_lambda=8),
]
best_spec, best_cv = None, -1.0
t0 = time.time()
for spec in grid:
    m = make_model(spec).set_params(random_state=SEED)
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    auc = roc_auc_score(yv, m.predict_proba(Xv)[:, 1])
    print(f"Info: cv_auc={auc:.4f} spec={spec}")
    if auc > best_cv:
        best_cv, best_spec = auc, spec
print(f"Search time: {time.time() - t0:.1f}s best_cv={best_cv:.4f}")

# --- final: bag the winning spec ------------------------------------------------
models = []
for seed in (42, 7, 13, 2025, 314):
    m = make_model(best_spec).set_params(random_state=seed)
    t0 = time.time()
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s x{len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
