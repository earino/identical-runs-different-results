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

# --- feature spec (fitted on train only) --------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _int_from_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["Month_i"] = _int_from_c(X["Month"].astype(str))
    X["Day_i"] = _int_from_c(X["DayofMonth"].astype(str))
    X["Dow_i"] = _int_from_c(X["DayOfWeek"].astype(str))
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_min"] = (dep // 100) * 60 + (dep % 100)
    X["dep_min"] = X["dep_min"].where(X["dep_min"] <= 24 * 60 - 1)
    ang = 2 * np.pi * X["dep_min"] / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- config sweep (selection on eval AUC; final model = best config) ----------
GRID = [
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=1, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=10, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=50, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=10, gamma=2.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=10, gamma=10.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=10, gamma=0.0, reg_lambda=10.0, subsample=1.0),
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=10, gamma=0.0, reg_lambda=1.0, subsample=0.8),
    dict(n_estimators=300, max_depth=6, learning_rate=0.1, min_child_weight=10, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=300, max_depth=6, learning_rate=0.1, min_child_weight=50, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=300, max_depth=4, learning_rate=0.1, min_child_weight=10, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=300, max_depth=4, learning_rate=0.1, min_child_weight=50, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=300, max_depth=8, learning_rate=0.1, min_child_weight=50, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_weight=10, gamma=0.0, reg_lambda=1.0, subsample=1.0),
    dict(n_estimators=50, max_depth=6, learning_rate=0.1, min_child_weight=10, gamma=0.0, reg_lambda=1.0, subsample=1.0),
]

Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

t0 = time.time()
best = None
for i, cfg in enumerate(GRID):
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **cfg
    )
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    print(f"cfg {i}: {auc:.4f} {cfg}")
    if best is None or auc > best[1]:
        best = (cfg, auc)
print(f"Training time: {time.time() - t0:.1f}s")
model = xgb.XGBClassifier(
    tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **best[0]
)
model.fit(Xtr, ytr)
print(f"best cfg: {best[0]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
