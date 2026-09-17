"""XGBoost ensemble for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

FEAT_NUM = ["DepHour", "DepMinute", "Distance"]
FEAT_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "CarrierHour"]
feature_cols = FEAT_NUM + FEAT_CAT


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering only (no fitted statistics)."""
    X = df.copy()
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["DepHour"] = (dep // 100) % 24
    X["DepMinute"] = dep % 100
    X["CarrierHour"] = X["UniqueCarrier"].astype(str) + "_" + X["DepHour"].astype(str)
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


# --- model: small ensemble of diverse XGBoost configs --------------------------
CONFIGS = [
    dict(n_estimators=800, max_depth=5, learning_rate=0.04, colsample_bylevel=0.5),
    dict(n_estimators=800, max_depth=5, learning_rate=0.04, reg_alpha=1.0),
    dict(n_estimators=600, max_depth=6, learning_rate=0.05),
    dict(n_estimators=300, max_depth=6, learning_rate=0.1),
    dict(n_estimators=800, max_depth=5, learning_rate=0.04, colsample_bylevel=0.3),
]

models = [
    xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **cfg,
    )
    for cfg in CONFIGS
]

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    preds = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
