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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000 and c not in ("Month", "DayofMonth", "DayOfWeek")]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
# group rare categorical levels (fitted on train only) to reduce cross-year overfitting
RARE_MIN = 400
cat_levels, rare_keep = {}, {}
for c in cat_cols:
    vc = train[c].value_counts()
    keep = vc[vc >= RARE_MIN].index
    rare_keep[c] = keep
    cat_levels[c] = pd.Index(sorted(keep) + ["__RARE__"])

# frequency maps fitted on training data only
FREQ = {c: train[c].value_counts() for c in ["Origin", "Dest", "UniqueCarrier"]}
_tr_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
FREQ["route"] = _tr_route.value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c, nc in [("Month", "month_n"), ("DayofMonth", "dom_n"), ("DayOfWeek", "dow_n")]:
        X[nc] = df[c].astype(str).str.replace("c-", "", regex=False).astype(float)
    for c in ["Origin", "Dest", "UniqueCarrier"]:
        X[c + "_freq"] = df[c].map(FREQ[c]).fillna(0).astype(float)
    _route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_freq"] = _route.map(FREQ["route"]).fillna(0).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c].where(X[c].isin(rare_keep[c]), "__RARE__"), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed-averaged ensemble --------------------------------------------
X_train, y_train = prepare(train), to_y(train)
t0 = time.time()
models = []
for s_ in [42, 7, 123, 2024, 99]:
    m = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        random_state=s_,
        n_jobs=N_JOBS,
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
