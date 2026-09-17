"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
RAW_COLS = [c for c in RAW_COLS if c not in obj_cols or c in cat_cols]
TIME_COLS = ["dep_hour", "dep_min", "dep_minutes"]
feature_cols = RAW_COLS + TIME_COLS
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[RAW_COLS].copy()
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["dep_hour"] = (dep // 100).clip(0, 23)
    X["dep_min"] = dep % 100
    X["dep_minutes"] = X["dep_hour"] * 60 + X["dep_min"]
    X = X[feature_cols]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# Pick the number of trees on an internal validation split (never on eval.csv, to avoid leakage).
X_all, y_all = prepare(train), to_y(train)
X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all)
probe = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **PARAMS)
t0 = time.time()
probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_rounds = int(probe.best_iteration) + 1
print(f"Probe time: {time.time() - t0:.1f}s  best_rounds={best_rounds}  valid_auc={probe.best_score:.4f}")

# Refit on all training data with the selected capacity.
model = xgb.XGBClassifier(n_estimators=int(best_rounds * 1.15) + 1, **PARAMS)
t0 = time.time()
model.fit(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
