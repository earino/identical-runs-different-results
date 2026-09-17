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

# frequency (count) encodings fitted on training data only
_route_tr = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str))
_freq_maps = {
    "freq_route": _route_tr.value_counts(),
    "freq_origin": train["Origin"].astype(str).value_counts(),
    "freq_dest": train["Dest"].astype(str).value_counts(),
    "freq_carrier": train["UniqueCarrier"].astype(str).value_counts(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # scheduled departure time: hhmm integer -> continuous time-of-day
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_tod"] = ((dep // 100).clip(0, 23) + (dep % 100) / 60.0).to_numpy()
    # count encodings
    route = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    X["freq_route"] = pd.Series(route).map(_freq_maps["freq_route"]).fillna(0).to_numpy()
    X["freq_origin"] = df["Origin"].astype(str).map(_freq_maps["freq_origin"]).fillna(0).to_numpy()
    X["freq_dest"] = df["Dest"].astype(str).map(_freq_maps["freq_dest"]).fillna(0).to_numpy()
    X["freq_carrier"] = df["UniqueCarrier"].astype(str).map(_freq_maps["freq_carrier"]).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 5
models = [
    xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    for i in range(N_MODELS)
]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
for m in models:
    m.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
