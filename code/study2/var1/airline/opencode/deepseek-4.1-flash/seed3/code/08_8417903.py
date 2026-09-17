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


def _tmin(df: pd.DataFrame) -> pd.Series:
    dep = df["DepTime"].astype(int)
    return ((dep // 100).clip(0, 23) * 60 + (dep % 100).clip(0, 59))


_ch_train = train["UniqueCarrier"].astype(str) + "_" + (_tmin(train) // 30).astype(str)
carrier_hour_levels = pd.Index(sorted(_ch_train.unique()))

# --- target encoding (fit on training data only, out-of-fold for training rows) --
TE_COLS = ["UniqueCarrier", "Origin", "Dest"]
SMOOTH = 20.0
_yv = (train[TARGET] == POSITIVE).astype(float)
_prior = float(_yv.mean())
te_map = {}
for c in TE_COLS:
    g = _yv.groupby(train[c].values).agg(["sum", "count"])
    te_map[c] = ((g["sum"] + SMOOTH * _prior) / (g["count"] + SMOOTH)).to_dict()

from sklearn.model_selection import KFold
_oof = {}
_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for c in TE_COLS:
    enc = np.full(len(train), _prior, dtype=float)
    for tr_idx, va_idx in _kf.split(train):
        p = float(_yv.iloc[tr_idx].mean())
        g = _yv.iloc[tr_idx].groupby(train[c].iloc[tr_idx].values).agg(["sum", "count"])
        mp = ((g["sum"] + SMOOTH * p) / (g["count"] + SMOOTH)).to_dict()
        enc[va_idx] = train[c].iloc[va_idx].map(mp).fillna(p).values
    _oof[c + "_te"] = enc


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in TE_COLS:
        X[c + "_te"] = df[c].map(te_map[c]).fillna(_prior).astype(float)
    _ch = df["UniqueCarrier"].astype(str) + "_" + (_tmin(df) // 30).astype(str)
    X["carrier_hour"] = pd.Categorical(_ch, categories=carrier_hour_levels)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_train = prepare(train)
for c in TE_COLS:
    X_train[c + "_te"] = _oof[c + "_te"]
y_train = to_y(train)

SEEDS = [42, 7, 2024, 1337, 99]
models = []
t0 = time.time()
for s in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=13,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
