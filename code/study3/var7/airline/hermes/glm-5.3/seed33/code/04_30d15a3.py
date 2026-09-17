"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# train-only statistics: how many train flights depart in this (carrier/origin/dest, hour) block
_h_tr = (train["DepTime"] // 100) % 24
SCHED_CNT = {}
for _c in ["UniqueCarrier", "Origin", "Dest"]:
    _k = train[_c].astype(str) + "|" + _h_tr.astype(str)
    SCHED_CNT[_c] = _k.value_counts()


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # numeric + cyclical time features
    mins = (X["DepTime"] // 100) * 60 + X["DepTime"] % 100
    X["dep_minutes"] = mins
    X["dep_hour"] = X["DepTime"] // 100
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["month_n"] = _cnum(X["Month"])
    X["dom_n"] = _cnum(X["DayofMonth"])
    X["dow_n"] = _cnum(X["DayOfWeek"])
    # schedule-density counts (train-only stats)
    h = (X["DepTime"] // 100) % 24
    for key_col, name in [("UniqueCarrier", "ch"), ("Origin", "oh"), ("Dest", "dh")]:
        key = df[key_col].astype(str) + "|" + h.astype(str)
        X[f"sched_cnt_{name}"] = np.log1p(key.map(SCHED_CNT[key_col]).fillna(0))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of XGBoost models over a range of depths: shallow trees transfer better
# across the 2005->2006 time shift; averaging over depths adds robustness.
MEMBERS = [dict(max_depth=d, n_estimators=300, learning_rate=0.05) for d in range(1, 8)]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for i, cfg in enumerate(MEMBERS):
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  members={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    out = np.zeros(len(X))
    for m in models:
        out += m.predict_proba(X)[:, 1] / len(models)
    return out


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
