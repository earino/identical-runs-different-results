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

# --- extra feature engineering (all fitted on train only) ----------------------
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
ROUTE_LEVELS = pd.Index(sorted(_route.unique()))
_dep_min = (train["DepTime"] // 100) * 60 + (train["DepTime"] % 100)
DEP15_LEVELS = pd.Index(sorted((_dep_min // 15).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].to_numpy()
    mins = (dep // 100) * 60 + (dep % 100)
    X["dep_minutes"] = mins
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["dep_bin15"] = pd.Categorical(mins // 15, categories=DEP15_LEVELS)
    X["route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=ROUTE_LEVELS
    )
    X["month_num"] = df["Month"].str.slice(2).astype(int)
    X["dom_num"] = df["DayofMonth"].str.slice(2).astype(int)
    X["dow_num"] = df["DayOfWeek"].str.slice(2).astype(int)
    X["log_distance"] = np.log1p(df["Distance"].to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
