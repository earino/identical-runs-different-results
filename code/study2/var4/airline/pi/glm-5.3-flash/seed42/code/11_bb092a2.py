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
# train-quintile edges for Distance (structural, no labels)
DIST_EDGES = np.unique(np.quantile(pd.to_numeric(train["Distance"], errors="coerce").dropna(), [0, .2, .4, .6, .8, 1.0]))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # DepTime (scheduled departure hhmm) -> time-of-day features
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100) % 24
    minutes = hour * 60 + (dt % 100)
    ang = 2 * np.pi * minutes / 1440.0
    X["DepHour"] = hour
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    carr = df["UniqueCarrier"].astype(str)
    h_str = hour.fillna(-1).astype(int).astype(str)
    X["CarrierHour"] = pd.Categorical(carr + "|h" + h_str,
        categories=(train["UniqueCarrier"].astype(str) + "|h" + ((pd.to_numeric(train["DepTime"], errors="coerce") // 100 % 24).fillna(-1).astype(int).astype(str))).unique())
    X["CarrierMonth"] = pd.Categorical(carr + "|m" + df["Month"].astype(str),
        categories=(train["UniqueCarrier"].astype(str) + "|m" + train["Month"].astype(str)).unique())
    dbin = pd.cut(pd.to_numeric(df["Distance"], errors="coerce"), bins=DIST_EDGES,
                  labels=list(range(len(DIST_EDGES) - 1)), include_lowest=True).astype(str)
    X["CarrierDist"] = pd.Categorical(carr + "|d" + dbin,
        categories=(train["UniqueCarrier"].astype(str) + "|d" + pd.cut(pd.to_numeric(train["Distance"], errors="coerce"), bins=DIST_EDGES, labels=list(range(len(DIST_EDGES) - 1)), include_lowest=True).astype(str)).unique())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int, **kw) -> xgb.XGBClassifier:
    params = dict(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                  enable_categorical=True, random_state=seed, n_jobs=N_JOBS)
    params.update(kw)
    return xgb.XGBClassifier(**params)

X_full, y_full = prepare(train), to_y(train)
t0 = time.time()
MEMBERS = [
    make_model(42),
    make_model(7, subsample=0.7),
    make_model(13, colsample_bytree=0.7),
    make_model(99, n_estimators=400, learning_rate=0.05, subsample=0.8),
    make_model(3, subsample=0.85),
    make_model(5, colsample_bytree=0.8),
    make_model(11, subsample=0.9),
    make_model(17, colsample_bytree=0.9),
    make_model(23, n_estimators=300, learning_rate=0.08),
    make_model(29, subsample=0.95),
    make_model(31, colsample_bytree=0.95),
    make_model(37, subsample=0.8, colsample_bytree=0.9),
    make_model(41, n_estimators=500, learning_rate=0.04),
]
for m in MEMBERS:
    m.fit(X_full, y_full)
print(f"Training time: {time.time() - t0:.1f}s ({len(MEMBERS)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in MEMBERS], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
