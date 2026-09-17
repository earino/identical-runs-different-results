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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering ------------------------------------------------------
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]

# Columns dropped because their year-specific patterns do not transfer (train=2005, eval/holdout=2006).
DROP = ["Month", "DayofMonth"]

CAT_FEATURES = [c for c in ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"] if c not in DROP]
NUM_FEATURES = ["DepTime", "dep_hour", "dep_min", "is_weekend", "Distance"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = df[RAW_COLS].copy()
    dep = pd.to_numeric(X["DepTime"], errors="coerce").fillna(0).astype(int)
    X["dep_hour"] = (dep // 100).clip(0, 23)
    X["dep_min"] = (dep % 100).clip(0, 59)
    X["is_weekend"] = X["DayOfWeek"].isin(["c-6", "c-7"]).astype(int)
    return X


_eng_train = engineer(train)
cat_levels = {c: pd.Index(sorted(_eng_train[c].dropna().unique())) for c in CAT_FEATURES}
feature_cols = NUM_FEATURES + CAT_FEATURES


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = engineer(df)
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr, ytr = prepare(train), to_y(train)

# Small ensemble of XGBoost models: averaging decorrelated seeds/configs reduces variance
# and generalizes better across the 2005->2006 distribution shift than any single model.
CONFIGS = [
    dict(max_depth=6, subsample=0.8, colsample_bytree=0.8, seed=42),
    dict(max_depth=6, subsample=0.8, colsample_bytree=0.8, seed=7),
    dict(max_depth=6, subsample=0.7, colsample_bytree=0.7, seed=2024),
    dict(max_depth=5, subsample=0.8, colsample_bytree=0.8, seed=11),
    dict(max_depth=7, subsample=0.8, colsample_bytree=0.8, seed=123),
    dict(max_depth=6, subsample=0.9, colsample_bytree=0.6, seed=99),
]

t0 = time.time()
models = []
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=cfg["max_depth"],
        learning_rate=0.05,
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
