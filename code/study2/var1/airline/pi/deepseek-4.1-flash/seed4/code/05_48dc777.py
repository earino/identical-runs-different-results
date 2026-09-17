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

# --- feature setup ------------------------------------------------------------
raw_feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in raw_feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
base_feature_cols = [c for c in raw_feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

ENG = ["dep_hour", "dep_min", "dep_frac", "is_night", "log_distance", "is_weekend", "dom", "dow"]
feature_cols = base_feature_cols + ENG


def _num(s: pd.Series) -> pd.Series:
    """c-<n> string -> int, plain numeric passes through."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[base_feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].astype(int).to_numpy()
    hour = (dep // 100) % 24
    minute = dep % 100
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["dep_frac"] = (hour * 60 + minute) / 1440.0
    X["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)
    X["log_distance"] = np.log1p(df["Distance"].astype(float).to_numpy())
    X["dom"] = _num(df["DayofMonth"]).to_numpy()
    X["dow"] = _num(df["DayOfWeek"]).to_numpy()
    X["is_weekend"] = (X["dow"] >= 6).astype(int)
    return X[feature_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of XGBoost models ---------------------------------
X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
CFGS = [
    dict(max_depth=d, learning_rate=lr, subsample=ss, colsample_bytree=cs,
         min_child_weight=mcw, n_estimators=400)
    for d, lr, ss, cs, mcw in [
        (4, 0.05, 0.9, 0.8, 1),
        (5, 0.05, 0.9, 0.8, 1),
        (6, 0.05, 0.9, 0.8, 5),
        (5, 0.05, 0.8, 0.6, 5),
        (5, 0.05, 0.7, 0.5, 10),
        (4, 0.05, 0.8, 0.6, 5),
        (6, 0.05, 0.8, 0.5, 10),
        (5, 0.03, 0.9, 0.7, 1),
        (4, 0.03, 0.9, 0.7, 5),
        (6, 0.03, 0.9, 0.7, 10),
        (5, 0.05, 0.7, 0.8, 20),
        (4, 0.05, 0.7, 0.5, 10),
    ]
]
for i, cfg in enumerate(CFGS):
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 101 * i,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
