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
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_hour"] = (dep // 100).clip(0, 23)
    X["dep_min"] = dep % 100
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["is_weekend"] = dow.isin([6, 7]).astype(int)
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["doy"] = (month - 1) * 30.44 + dom
    X["hour_sin"] = np.sin(2 * np.pi * X["dep_hour"] / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * X["dep_hour"] / 24.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of diverse XGBoost models --------------------------------
PARAMS = [
    dict(n_estimators=380, max_depth=4, learning_rate=0.04, min_child_weight=1),
    dict(n_estimators=300, max_depth=5, learning_rate=0.04, min_child_weight=4, colsample_bytree=0.8),
    dict(n_estimators=150, max_depth=6, learning_rate=0.05, min_child_weight=10),
    dict(n_estimators=600, max_depth=3, learning_rate=0.05, min_child_weight=1),
    dict(n_estimators=500, max_depth=4, learning_rate=0.03, min_child_weight=2, subsample=0.8),
    dict(n_estimators=380, max_depth=4, learning_rate=0.04, min_child_weight=1, max_cat_to_onehot=1),
    dict(n_estimators=380, max_depth=4, learning_rate=0.04, min_child_weight=1, max_cat_threshold=16),
]

X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for i, p in enumerate(PARAMS):
    for s in range(2):
        m = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED + 10 * i + s,
            n_jobs=N_JOBS,
            **p,
        )
        m.fit(X_train, y_train)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
