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
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepHour"] = (dep // 100).clip(0, 23)
    X["DepMinute"] = dep % 100
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["IsWeekend"] = (dow >= 6).astype(int)
    mon = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["DayOfYear"] = (mon - 1) * 31 + dom
    X["HourXWeekend"] = X["DepHour"] * X["IsWeekend"]
    X["HourXDist"] = X["DepHour"] * X["Distance"] / 1000.0
    X["HourFromNoonSq"] = (X["DepHour"] - 13).astype(float) ** 2
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    learning_rate=0.03,
    min_child_weight=15,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
MODEL_SPECS = [
    dict(n_estimators=500, max_depth=5, random_state=SEED + 100 * s)
    if d == 5 else
    dict(n_estimators=400, max_depth=6, random_state=SEED + 100 * s)
    if d == 6 else
    dict(n_estimators=300, max_depth=7, random_state=SEED + 100 * s)
    for s in range(4)
    for d in (5, 6, 7)
]

X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for spec in MODEL_SPECS:
    m = xgb.XGBClassifier(**{**BASE, **spec})
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return preds


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
