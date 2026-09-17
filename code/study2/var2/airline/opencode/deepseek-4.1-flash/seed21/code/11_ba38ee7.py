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


def _route(d: pd.DataFrame) -> pd.Series:
    return d["Origin"].astype(str) + "_" + d["Dest"].astype(str)


COUNT_SPECS = {
    "cnt_carrier": lambda d: d["UniqueCarrier"],
    "cnt_origin": lambda d: d["Origin"],
    "cnt_dest": lambda d: d["Dest"],
    "cnt_route": _route,
}
COUNT_MAPS = {n: fn(train).value_counts() for n, fn in COUNT_SPECS.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str.replace("c-", "", regex=False).astype(float)
    X["DayofMonth"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(float)
    X["DayOfWeek"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(float)
    dt = df["DepTime"].to_numpy()
    hour = (dt // 100) % 24
    minute = np.where(dt % 100 < 60, dt % 100, 0)
    mod = hour * 60 + minute
    X["dep_mod"] = mod
    X["dep_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    X["Distance"] = df["Distance"].to_numpy()
    X["log_dist"] = np.log1p(df["Distance"].to_numpy())
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for n, fn in COUNT_SPECS.items():
        X[n] = fn(df).map(COUNT_MAPS[n]).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
SEEDS = [42, 7, 2024]
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.005,
        subsample=0.8,
        colsample_bytree=0.3,
        min_child_weight=100,
        reg_lambda=20.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
