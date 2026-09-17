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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# congestion lookups fit on train only
_hr = pd.to_numeric(train["DepTime"], errors="coerce") // 100
oh = train.groupby(["Origin", _hr], observed=True).size().to_dict()
dh = train.groupby(["Dest", _hr], observed=True).size().to_dict()
ch = train.groupby(["UniqueCarrier", _hr], observed=True).size().to_dict()
oc = train["Origin"].value_counts().to_dict()
dc = train["Dest"].value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    hr = pd.to_numeric(X["DepTime"], errors="coerce") // 100
    X["hour_num"] = hr.astype("float64")
    X["orig_hour_count"] = [oh.get((o, h), 0.0) for o, h in zip(X["Origin"], hr)]
    X["dest_hour_count"] = [dh.get((d, h), 0.0) for d, h in zip(X["Dest"], hr)]
    X["carrier_hour_count"] = [ch.get((c, h), 0.0) for c, h in zip(X["UniqueCarrier"], hr)]
    X["orig_count"] = X["Origin"].map(oc).fillna(0.0).astype("float64")
    X["dest_count"] = X["Dest"].map(dc).fillna(0.0).astype("float64")
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _ranks(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    r = np.empty_like(order, dtype=float)
    r[order] = np.arange(1, len(p) + 1)
    return r / (len(p) + 1.0)


# --- model --------------------------------------------------------------------
Xtr = prepare(train)
ytr = to_y(train)
FAMILIES = [
    dict(n_estimators=150, max_depth=5, learning_rate=0.08, colsample_bytree=0.4),
    dict(n_estimators=150, max_leaves=24, grow_policy="lossguide", learning_rate=0.08, colsample_bytree=0.7),
]
models = []
t0 = time.time()
for fam in FAMILIES:
    for seed in range(12):
        m = xgb.XGBClassifier(
            **fam,
            subsample=0.7,
            min_child_weight=20,
            reg_lambda=1.0,
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=N_JOBS,
        )
        m.fit(Xtr, ytr)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.stack([_ranks(m.predict_proba(X)[:, 1]) for m in models])
    return ps.mean(axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
