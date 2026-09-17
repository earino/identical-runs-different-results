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
ORDINAL_COLS = ["Month", "DayofMonth", "DayOfWeek"]
cat_cols = [c for c in cat_cols if c not in ORDINAL_COLS]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in ORDINAL_COLS:
        X[c] = X[c].astype(str).str.replace("c-", "", regex=False).astype(float)
    dep = df["DepTime"].to_numpy()
    X["dep_hour"] = np.clip(dep // 100, 0, 23)
    X["dep_min"] = np.where(dep % 100 < 60, dep % 100, 0)
    X["dep_minutes"] = X["dep_hour"] * 60 + X["dep_min"]
    X["day_of_year"] = (X["Month"] - 1) * 30 + X["DayofMonth"]
    X["is_weekend"] = (X["DayOfWeek"] >= 6).astype(int)
    X = X.drop(columns=["DepTime", "Month", "DayofMonth", "DayOfWeek"])
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble model -----------------------------------------------------------
def make_model(depth: int, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=depth,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


MODEL_SPECS = [(3, SEED), (4, SEED), (5, SEED)]

X_train, y_train = prepare(train), to_y(train)
t0 = time.time()
models = []
for depth, seed in MODEL_SPECS:
    m = make_model(depth, seed)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return preds


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
