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
_FREQ = {
    "origin_freq": train["Origin"].value_counts().to_dict(),
    "dest_freq": train["Dest"].value_counts().to_dict(),
    "route_freq": (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts().to_dict(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["origin_freq"] = df["Origin"].map(_FREQ["origin_freq"]).fillna(0).astype(float)
    X["dest_freq"] = df["Dest"].map(_FREQ["dest_freq"]).fillna(0).astype(float)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_freq"] = route.map(_FREQ["route_freq"]).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
MODEL_SPECS = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42),
    dict(n_estimators=250, max_depth=4, learning_rate=0.05, min_child_weight=8, subsample=0.7, colsample_bytree=0.6, random_state=7),
    dict(n_estimators=300, max_depth=5, learning_rate=0.04, min_child_weight=15, subsample=0.8, colsample_bytree=0.7, random_state=123),
]

X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for spec in MODEL_SPECS:
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **spec,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
