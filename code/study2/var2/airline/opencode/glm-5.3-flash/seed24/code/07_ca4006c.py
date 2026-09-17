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

# counts fit on TRAIN only (drift-robust frequency encodings)
_route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_freq = {
    "Origin": train["Origin"].value_counts(),
    "Dest": train["Dest"].value_counts(),
    "Route": _route_train.value_counts(),
}
HOUR_LEVELS = pd.Index([str(i) for i in range(25)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = pd.to_numeric(X.pop("DepTime"), errors="coerce")
    mins = ((dt // 100) * 60 + (dt % 100)) % 1440  # 2400/2620 quirks -> wrap into the day
    X["hour"] = mins // 60
    X["minute"] = mins % 60
    ang = 2.0 * np.pi * mins / 1440.0
    X["tsin"] = np.sin(ang)
    X["tcos"] = np.cos(ang)
    X["logdist"] = np.log1p(pd.to_numeric(X.pop("Distance"), errors="coerce"))
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["freq_Origin"] = np.log1p(df["Origin"].map(_freq["Origin"]).fillna(0.0))
    X["freq_Dest"] = np.log1p(df["Dest"].map(_freq["Dest"]).fillna(0.0))
    X["freq_Route"] = np.log1p(route.map(_freq["Route"]).fillna(0.0))
    X["hour_cat"] = pd.Categorical((mins // 60).astype(int).astype(str), categories=HOUR_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(n_estimators=300, max_depth=5, learning_rate=0.1, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
CONFIGS = [
    dict(max_depth=5, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, subsample=1.0, colsample_bytree=0.7),
    dict(max_depth=6, subsample=0.9, colsample_bytree=0.9),
    dict(max_depth=6, subsample=0.7, colsample_bytree=1.0),
    dict(max_depth=5, subsample=0.9, colsample_bytree=0.9),
]


def make_model(cfg: dict, i: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**{**BASE, "random_state": SEED + i, **cfg})


t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
ENSEMBLE = [make_model(c, i) for i, c in enumerate(CONFIGS)]
for m in ENSEMBLE:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    p = np.mean([m.predict_proba(prepare(df))[:, 1] for m in ENSEMBLE], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
