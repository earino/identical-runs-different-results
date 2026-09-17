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
from sklearn.model_selection import StratifiedKFold

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
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].to_numpy(dtype=float)
    hh = dep // 100.0
    mm = dep % 100.0
    tot = hh * 60.0 + mm
    X["dep_min"] = tot
    X["dep_hour"] = hh
    X["dep_sin"] = np.sin(2.0 * np.pi * tot / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * tot / 1440.0)
    dow = df["DayOfWeek"].astype(str)
    X["is_weekend"] = dow.isin(["c-6", "c-7"]).astype(int)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CANDIDATES = [
    dict(n_estimators=30, max_depth=6, learning_rate=0.1),
    dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9),
    dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=5, reg_lambda=5),
    dict(n_estimators=100, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=10, reg_lambda=10),
    dict(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9),
    dict(n_estimators=60, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
         min_child_weight=3, reg_lambda=3),
    dict(n_estimators=150, max_depth=7, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85,
         min_child_weight=5, reg_lambda=2),
    dict(n_estimators=50, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
         min_child_weight=10, reg_lambda=10),
]


def make_model(cfg):
    return xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                             n_jobs=N_JOBS, **cfg)


Xtr = prepare(train)
ytr = to_y(train)

t0 = time.time()
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
best_cfg, best_score = None, -1.0
for cfg in CANDIDATES:
    scores = []
    for tr_idx, va_idx in skf.split(Xtr, ytr):
        m = make_model(cfg)
        m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        scores.append(roc_auc_score(ytr[va_idx], m.predict_proba(Xtr.iloc[va_idx])[:, 1]))
    sc = float(np.mean(scores))
    print(f"CV {sc:.4f} {cfg}")
    if sc > best_score:
        best_score, best_cfg = sc, cfg
print(f"Best CV {best_score:.4f} {best_cfg}")

model = make_model(best_cfg)
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
