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

# --- fitted artifacts (fit on TRAIN ONLY, applied inside prepare()) -----------
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _num_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


feature_cols = [
    "dep_min", "dep_sin", "dep_cos", "distance_log",
    "Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance",
    "UniqueCarrier", "Origin", "Dest",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # cyclical time of day; DepTime can exceed 2359 (late-night scheduled flights)
    hh = (df["DepTime"] // 100).astype(float)
    mm = (df["DepTime"] % 100).astype(float)
    dep_min = (hh * 60 + mm) % 1440
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(2 * np.pi * dep_min / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * dep_min / 1440)
    X["distance_log"] = np.log1p(df["Distance"].astype(float))
    X["Month"] = _num_col(df["Month"])
    X["DayofMonth"] = _num_col(df["DayofMonth"])
    X["DayOfWeek"] = _num_col(df["DayOfWeek"])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    reg_alpha=1.0,
)

CONFIGS = [
    dict(n_estimators=400, max_depth=6, min_child_weight=20, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=400, max_depth=6, min_child_weight=20, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=400, max_depth=6, min_child_weight=20, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=500, max_depth=8, min_child_weight=50, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0),
    dict(n_estimators=500, max_depth=8, min_child_weight=50, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0),
    dict(n_estimators=800, max_depth=6, min_child_weight=20, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=800, max_depth=6, min_child_weight=20, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=600, max_depth=6, min_child_weight=20, learning_rate=0.06, subsample=0.6, colsample_bytree=0.5),
    dict(n_estimators=600, max_depth=6, min_child_weight=20, learning_rate=0.06, subsample=0.6, colsample_bytree=0.5),
    dict(n_estimators=800, max_depth=8, min_child_weight=50, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=800, max_depth=8, min_child_weight=50, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=800, max_depth=6, min_child_weight=20, learning_rate=0.06, subsample=0.5, colsample_bytree=0.4),
    dict(n_estimators=800, max_depth=6, min_child_weight=20, learning_rate=0.06, subsample=0.5, colsample_bytree=0.4),
]
X_tr = prepare(train)
y_tr = to_y(train)

models = []
t0 = time.time()
for i, cfg in enumerate(CONFIGS):
    m = xgb.XGBClassifier(random_state=SEED + i, **cfg, **PARAMS)
    m.fit(X_tr, y_tr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
