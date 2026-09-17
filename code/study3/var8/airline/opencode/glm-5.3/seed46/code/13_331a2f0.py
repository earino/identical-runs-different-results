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
from sklearn.model_selection import StratifiedKFold

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

def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target encoding (fit on train only; OOF values used for training rows) ---
TE_ALPHA = 10.0
TE_KEYS = []  # ablation: base TEs off (carrier/origin/dest/route)
_y = pd.Series(to_y(train), index=train.index)
_prior = float(_y.mean())


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


_train_route = _route(train)
_te_maps: dict = {}


def _te_map(key: pd.Series) -> pd.Series:
    g = pd.DataFrame({"k": key, "y": _y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + TE_ALPHA * _prior) / (g["count"] + TE_ALPHA)


for k in TE_KEYS:
    src = _train_route if k == "route" else train[k]
    _te_maps[k] = _te_map(src)

# frequency maps (train only)
_freq_maps = {
    "Origin": train["Origin"].value_counts(),
    "Dest": train["Dest"].value_counts(),
    "route": _train_route.value_counts(),
}

_oof = pd.DataFrame(index=train.index)
_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in _skf.split(train, _y):
    for k in TE_KEYS:
        src_tr = _train_route.iloc[tr_idx] if k == "route" else train[k].iloc[tr_idx]
        src_va = _train_route.iloc[va_idx] if k == "route" else train[k].iloc[va_idx]
        _oof.loc[_oof.index[va_idx], "te_" + k] = src_va.map(_te_map(src_tr)).fillna(_prior).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    # numeric versions of the c-<n> string columns (categories stay too)
    for c, name in (("Month", "month"), ("DayofMonth", "dom"), ("DayOfWeek", "dow")):
        X[name] = df[c].str.slice(2).astype(float)
    # time-of-day features from DepTime (hhmm, values past 2400 wrap to next day)
    dt = df["DepTime"].astype(float)
    X["hour"] = dt // 100
    X["mins"] = (dt // 100) % 24 * 60 + dt % 100
    ang = 2.0 * np.pi * X["mins"] / 1440.0
    X["sin_t"] = np.sin(ang)
    X["cos_t"] = np.cos(ang)
    # position within the hour (10-minute bucket; small but year-stable effect)
    X["m10"] = (dt % 100) // 10
    # seasonality (day-of-year angle) + distance transforms
    doy = (X["month"] - 1.0) * 30.4 + X["dom"]
    ang_y = 2.0 * np.pi * doy / 365.0
    X["sin_y"] = np.sin(ang_y)
    X["cos_y"] = np.cos(ang_y)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    # flight-frequency features (counts fit on train only)
    X["route"] = _route(df)
    for k in ("Origin", "Dest", "route"):
        X["freq_" + k] = np.log1p(X[k].map(_freq_maps[k]).fillna(0.0))
    # smoothed target encodings (maps fit on train only; unseen -> global prior)
    for k in TE_KEYS:
        X["te_" + k] = X[k].map(_te_maps[k]).fillna(_prior)
    X = X.drop(columns="route")
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def prepare_fit(df: pd.DataFrame) -> pd.DataFrame:
    # training rows get out-of-fold TE values instead of the (leaky) full-train maps
    X = prepare(df)
    if df is train and len(_oof.columns):
        X[list(_oof.columns)] = _oof
    return X


# --- model: seed/depth ensemble of XGBoost, each with ES on eval ----------------
X_tr, y_tr = prepare_fit(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)

models = []
t0 = time.time()
for i in range(16):
    depth = (6, 5, 7, 4, 6, 5, 7, 8, 6, 5, 7, 4, 6, 5, 7, 8)[i]
    if i % 2 == 0:  # current best family: aggressive sampling + gamma
        m = xgb.XGBClassifier(
            n_estimators=1500,
            max_depth=depth,
            learning_rate=0.05,
            subsample=0.5,
            colsample_bytree=0.5,
            colsample_bynode=0.7,
            gamma=0.5,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=100,
            random_state=SEED + i,
            n_jobs=N_JOBS,
        )
    else:  # alternate family: gentler sampling, slower lr
        m = xgb.XGBClassifier(
            n_estimators=3000,
            max_depth=depth,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=100,
            random_state=SEED + 100 + i,
            n_jobs=N_JOBS,
        )
    if i >= 12:  # third family: deep trees kept honest by heavy sampling + gamma
        m = xgb.XGBClassifier(
            n_estimators=1500,
            max_depth=9 + i % 2,
            learning_rate=0.05,
            subsample=0.4,
            colsample_bytree=0.4,
            gamma=1.0,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=100,
            random_state=SEED + 200 + i,
            n_jobs=N_JOBS,
        )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  best_iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
