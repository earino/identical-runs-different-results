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

# --- static feature bookkeeping (fit on training data only) -------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _hours(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    t = np.where(t >= 2400, t - 2400, t)
    return pd.Series((t // 100).astype(int), index=df.index)


def _cnt_key(df: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "origin_hour":
        return df["Origin"].astype(str) + "|" + _hours(df).astype(str)
    if kind == "dest_hour":
        return df["Dest"].astype(str) + "|" + _hours(df).astype(str)
    if kind == "carrier_hour":
        return df["UniqueCarrier"].astype(str) + "|" + _hours(df).astype(str)
    if kind == "origin_dow":
        return df["Origin"].astype(str) + "|" + df["DayOfWeek"].astype(str)
    raise ValueError(kind)


CNT_KINDS = ["origin_hour", "dest_hour", "carrier_hour", "origin_dow"]
cnt_maps = {k: _cnt_key(train, k).value_counts(normalize=True).to_dict() for k in CNT_KINDS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hours = _hours(df)
    X["dep_hour_cat"] = pd.Categorical(hours, categories=list(range(24)))
    X["dep_minute"] = (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) % 100).astype(float)
    for k in CNT_KINDS:
        X[f"cnt_{k}"] = _cnt_key(df, k).map(cnt_maps[k]).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_all = to_y(train)
X_tr = prepare(train)

# --- model ensemble -----------------------------------------------------------
CONFIGS = []
for sd in (1, 2, 3, 4):
    CONFIGS += [
        dict(max_depth=3, n_estimators=200, learning_rate=0.1, seed=sd, colsample=0.8, subsample=0.8),
        dict(max_depth=4, n_estimators=120, learning_rate=0.1, seed=sd, colsample=0.7, subsample=0.85),
        dict(max_depth=5, n_estimators=80, learning_rate=0.08, seed=sd, colsample=0.7, subsample=0.85),
        dict(max_depth=6, n_estimators=60, learning_rate=0.08, seed=sd, colsample=0.7, subsample=0.85),
    ]
for sd in (1, 2):
    CONFIGS += [
        dict(max_depth=2, n_estimators=300, learning_rate=0.1, seed=sd, colsample=0.8, subsample=0.9),
        dict(max_depth=7, n_estimators=50, learning_rate=0.06, seed=sd, colsample=0.7, subsample=0.85),
    ]

models = []
t0 = time.time()
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample"],
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
