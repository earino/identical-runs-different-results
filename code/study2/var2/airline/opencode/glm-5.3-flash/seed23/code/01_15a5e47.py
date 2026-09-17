"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

FEAT_NUM = ["Month", "DayofMonth", "DayOfWeek", "DepHour", "DepMinute", "Distance"]
FEAT_CAT = ["UniqueCarrier", "Origin", "Dest", "Route"]
feature_cols = FEAT_NUM + FEAT_CAT


def _int_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering only (no fitted statistics)."""
    X = df.copy()
    X["Month"] = _int_col(X["Month"])
    X["DayofMonth"] = _int_col(X["DayofMonth"])
    X["DayOfWeek"] = _int_col(X["DayOfWeek"])
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["DepHour"] = (dep // 100).clip(0, 23)
    X["DepMinute"] = (dep % 100).clip(0, 59)
    X["Route"] = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    return X


_tr_fe = add_base_features(train)
cat_levels = {c: pd.Index(sorted(_tr_fe[c].dropna().unique())) for c in FEAT_CAT}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = add_base_features(df)
    X = X[feature_cols].copy()
    for c in FEAT_CAT:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
