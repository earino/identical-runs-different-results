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
from sklearn.model_selection import KFold

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
    for c in TE_COLS:
        X["te_" + c] = te_key(c, df).map(te_full[c]).fillna(PRIOR).astype("float32")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- smoothed target encodings (train-only statistics) ------------------------

def _dep_hour(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype("int32")
    return (dep // 100).clip(0, 23)


TE_KEYS = {
    "Origin": lambda df: df["Origin"].astype(str),
    "Dest": lambda df: df["Dest"].astype(str),
    "UniqueCarrier": lambda df: df["UniqueCarrier"].astype(str),
    "Month": lambda df: df["Month"].astype(str),
    "DayOfWeek": lambda df: df["DayOfWeek"].astype(str),
    "route": lambda df: df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
    "org_hour": lambda df: df["Origin"].astype(str) + "_" + _dep_hour(df).astype(str),
    "carrier_org": lambda df: df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str),
}
TE_COLS = list(TE_KEYS)
SMOOTH = 20.0
PRIOR = float(to_y(train).mean())


def te_key(col: str, df: pd.DataFrame) -> pd.Series:
    return TE_KEYS[col](df)


def _te_map(df: pd.DataFrame, col: str) -> dict:
    y = to_y(df)
    g = pd.DataFrame({"c": te_key(col, df), "y": y}).groupby("c")["y"].agg(["mean", "count"])
    return ((g["mean"] * g["count"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)).to_dict()


te_full = {c: _te_map(train, c) for c in TE_COLS}


def add_oof_te(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold target encodings for the training rows (avoids self-leakage)."""
    y = to_y(df)
    for c in TE_COLS:
        s = te_key(c, df).to_numpy()
        oof = np.full(len(df), PRIOR, dtype="float32")
        for tr, va in KFold(5, shuffle=True, random_state=SEED).split(s):
            sub = pd.DataFrame({"c": s[tr], "y": y[tr]}).groupby("c")["y"].agg(["mean", "count"])
            m = (sub["mean"] * sub["count"] + PRIOR * SMOOTH) / (sub["count"] + SMOOTH)
            oof[va] = pd.Series(s[va]).map(m).fillna(PRIOR).to_numpy()
        X["te_" + c] = oof
    return X


# --- model --------------------------------------------------------------------
# small ensemble of XGBoost models differing in depth for variance reduction
CONFIGS = [(d, s) for d in (4, 5, 6) for s in (42, 7, 2024)]

t0 = time.time()
X_train = add_oof_te(prepare(train), train)
y_train = to_y(train)
models = []
for depth, seed in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=depth,
        colsample_bytree=0.6,
        tree_method="hist",
        max_bin=512,
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
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
