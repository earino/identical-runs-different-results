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

# --- interaction target encodings (fit on train only) ---------------------------
from sklearn.model_selection import KFold

PRIOR = float((train[TARGET] == POSITIVE).mean())
TE_M = 20.0
_hb = (train["DepTime"] // 100 % 24) // 3  # 3-hour bins


def _te_stats(keys: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.reset_index(drop=True), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * PRIOR) / (g["count"] + m)


_y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
TE_KEYS = {
    "origin_hb": train["Origin"].astype(str) + "_" + _hb.astype(str),
    "dest_hb": train["Dest"].astype(str) + "_" + _hb.astype(str),
    "carrier_hb": train["UniqueCarrier"].astype(str) + "_" + _hb.astype(str),
    "hb_dow": _hb.astype(str) + "_" + train["DayOfWeek"].astype(str),
}
FULL_TE = {name: _te_stats(ks, _y_train, TE_M) for name, ks in TE_KEYS.items()}
OOF_TE = {name: np.zeros(len(train)) for name in TE_KEYS}
for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
    for name, ks in TE_KEYS.items():
        te = _te_stats(ks.iloc[_tr], _y_train[_tr], TE_M)
        OOF_TE[name][_va] = ks.iloc[_va].map(te).fillna(PRIOR).to_numpy()
ROUTE_COUNT = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    hh = (X["DepTime"] // 100) % 24
    mm = X["DepTime"] % 100
    X["hour"] = hh
    X["minute"] = mm
    tod = hh + mm / 60.0
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    hb = ((hh // 3)).astype(str)
    X["origin_hb_te"] = (df["Origin"].astype(str) + "_" + hb).map(FULL_TE["origin_hb"]).fillna(PRIOR)
    X["dest_hb_te"] = (df["Dest"].astype(str) + "_" + hb).map(FULL_TE["dest_hb"]).fillna(PRIOR)
    X["carrier_hb_te"] = (df["UniqueCarrier"].astype(str) + "_" + hb).map(FULL_TE["carrier_hb"]).fillna(PRIOR)
    X["hb_dow_te"] = (hb + "_" + df["DayOfWeek"].astype(str)).map(FULL_TE["hb_dow"]).fillna(PRIOR)
    X["doy"] = (df["Month"].str.slice(2).astype(int) - 1) * 31 + df["DayofMonth"].str.slice(2).astype(int)
    X["route_count"] = np.log1p((df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(ROUTE_COUNT).fillna(0))
    X["log_dist"] = np.log1p(X["Distance"])
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=4, learning_rate=0.02, n_estimators=8000, early_stopping_rounds=300, random_state=42),
    dict(max_depth=6, learning_rate=0.03, n_estimators=8000, early_stopping_rounds=300, random_state=7),
    dict(max_depth=5, learning_rate=0.05, n_estimators=4000, early_stopping_rounds=150, random_state=2024),
    dict(max_depth=3, learning_rate=0.02, n_estimators=8000, early_stopping_rounds=300, random_state=13),
    dict(max_depth=4, learning_rate=0.05, n_estimators=4000, early_stopping_rounds=150, random_state=99,
         subsample=0.8, colsample_bytree=0.8),
]

t0 = time.time()
X_train = prepare(train)
for name in TE_KEYS:  # train rows get leak-free OOF target encodings
    X_train[name + "_te"] = OOF_TE[name]
X_eval, y_eval = prepare(evald), to_y(evald)
y_train = to_y(train)
models = []
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, eval_metric="auc", n_jobs=N_JOBS, **cfg
    )
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    auc_m = roc_auc_score(y_eval, m.predict_proba(X_eval)[:, 1])
    print(f"member d{cfg['max_depth']} lr{cfg['learning_rate']}: best_iter={m.best_iteration} auc={auc_m:.4f}")
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
