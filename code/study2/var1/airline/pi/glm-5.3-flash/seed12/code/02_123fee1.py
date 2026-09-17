"""XGBoost binary classifier with internal mini-grid model selection.

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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all = prepare(train)
y_all = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)

# --- model selection on a train-only validation split --------------------------------
GRID = [
    dict(max_depth=d, min_child_weight=w, subsample=s, colsample_bytree=0.9)
    for d in (4, 6, 8)
    for w in (1, 30)
    for s in (1.0, 0.7)
]

best = None
best_val = -1.0
t0 = time.time()
for i, params in enumerate(GRID):
    m = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=N_JOBS,
        **params,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    val_auc = roc_auc_score(y_val, m.predict_proba(X_val, iteration_range=(0, m.best_iteration + 1))[:, 1])
    print(f"grid[{i}] {params} -> val {val_auc:.4f} @ {m.best_iteration}")
    if val_auc > best_val:
        best_val, best = val_auc, (params, m.best_iteration)
print(f"grid time: {time.time() - t0:.1f}s  best={best} val={best_val:.4f}")

params, best_iter = best
model = xgb.XGBClassifier(
    n_estimators=best_iter + 1,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    **params,
)
t0 = time.time()
model.fit(X_all, y_all)
print(f"Final fit time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
