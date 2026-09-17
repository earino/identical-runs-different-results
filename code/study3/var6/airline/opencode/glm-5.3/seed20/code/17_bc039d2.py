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

# --- target encoding (fit on train only; OOF for train rows, smoothed full stats otherwise) -----
from sklearn.model_selection import KFold

y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_all.mean())
route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
TE_KEYS = {
    "te_carrier": train["UniqueCarrier"].astype(str),
    "te_origin": train["Origin"].astype(str),
    "te_dest": train["Dest"].astype(str),
    "te_route": route_train,
}
TE_M = {"te_carrier": 25, "te_origin": 50, "te_dest": 50, "te_route": 50}


def _te_stats(keys: pd.Series, y: np.ndarray, m: float):
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + PRIOR * m) / (g["count"] + m)).to_dict()


TE_FULL = {name: _te_stats(keys, y_all, m) for name, keys, m in ((n, k, TE_M[n]) for n, k in TE_KEYS.items())}
_oof = {name: np.zeros(len(train)) for name in TE_KEYS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    for name, keys in TE_KEYS.items():
        stats = _te_stats(keys.iloc[tr_idx], y_all[tr_idx], TE_M[name])
        _oof[name][va_idx] = keys.iloc[va_idx].map(stats).to_numpy()
TE_OOF = {name: pd.Series(v).fillna(PRIOR).to_numpy() for name, v in _oof.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["te_route"] = route.map(TE_FULL["te_route"]).fillna(PRIOR)
    X["te_carrier"] = df["UniqueCarrier"].astype(str).map(TE_FULL["te_carrier"]).fillna(PRIOR)
    X["te_origin"] = df["Origin"].astype(str).map(TE_FULL["te_origin"]).fillna(PRIOR)
    X["te_dest"] = df["Dest"].astype(str).map(TE_FULL["te_dest"]).fillna(PRIOR)
    X["doy"] = 30.0 * (df["Month"].str.slice(2).astype(float) - 1) + df["DayofMonth"].str.slice(2).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def prepare_train() -> pd.DataFrame:
    X = prepare(train)
    for name in TE_KEYS:
        X[name] = TE_OOF[name]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 10
models = []
t0 = time.time()
Xtr, Xev = prepare_train(), prepare(evald)
ytr, yev = to_y(train), to_y(evald)
for k in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=20,
        max_leaves=256,
        grow_policy="lossguide",
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.4,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        random_state=SEED + k,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s, best_iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
