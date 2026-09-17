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
# Month/DayofMonth encode year-specific seasonality that does not transfer across the
# 2005 -> 2006 time split; dropping them consistently improves eval AUC.
DROP_COLS = {"Month", "DayofMonth"}
feature_cols = [c for c in feature_cols if c not in DROP_COLS]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols if c in feature_cols}
cat_cols = [c for c in cat_cols if c in feature_cols]


def _dep_hour(s: pd.Series) -> pd.Series:
    d = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    return (d // 100).clip(0, 23).astype(int)


# schedule-density lookups fitted on training data only (frequency, not target statistics,
# so they are robust to the 2005 -> 2006 shift).
_oh_train = train["Origin"].astype(str) + "_" + _dep_hour(train["DepTime"]).astype(str)
_dh_train = train["Dest"].astype(str) + "_" + _dep_hour(train["DepTime"]).astype(str)
_ch_train = train["UniqueCarrier"].astype(str) + "_" + _dep_hour(train["DepTime"]).astype(str)
_oh_map = _oh_train.value_counts()
_dh_map = _dh_train.value_counts()
_ch_map = _ch_train.value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    tod = (hour * 60 + minute).astype(float)
    X["dep_hour"] = hour.astype(float)
    X["dep_min"] = minute.astype(float)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    oh = df["Origin"].astype(str) + "_" + hour.astype(str)
    dh = df["Dest"].astype(str) + "_" + hour.astype(str)
    ch = df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
    X["oh_cnt"] = oh.map(_oh_map).fillna(0).astype(float)
    X["dh_cnt"] = dh.map(_dh_map).fillna(0).astype(float)
    X["ch_cnt"] = ch.map(_ch_map).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Depth-diverse ensemble: shallow trees capture stable/low-order effects while deep trees
# capture interactions. Individually the deep models overfit the 2005 training year, but
# averaging predictions across a wide range of depths cancels much of that variance and
# transfers well to the later-year evaluation set. Deep members use fewer trees to stay
# within the per-experiment time budget.
_SCHEDULE = ([(d, 200) for d in range(2, 15)] + [(d, 120) for d in range(15, 18)]
             + [(d, 70) for d in range(18, 21)] + [(d, 40) for d in range(21, 25)]
             + [(d, 25) for d in range(25, 29)])
MODELS = [
    xgb.XGBClassifier(
        n_estimators=n,
        max_depth=d,
        learning_rate=0.12,
        colsample_bytree=0.4,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    for d, n in _SCHEDULE
]

_t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
for _m in MODELS:
    _m.fit(X_train, y_train)
print(f"Training time: {time.time() - _t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)
    return preds


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
