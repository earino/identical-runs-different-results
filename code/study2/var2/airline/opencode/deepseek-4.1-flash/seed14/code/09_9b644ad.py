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
# c-<n> encoded categorical columns (Month, DayofMonth, DayOfWeek): decode to int.
CYCLIC = {"Month": 12, "DayofMonth": 31, "DayOfWeek": 7}
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}


def _cint(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(float)


def _cnt_keys(df: pd.DataFrame) -> dict:
    dep = df["DepTime"].astype(float)
    h = (dep // 100).astype(int) % 24
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    return {
        "o_h": df["Origin"].astype(str) + "_" + h.astype(str),
        "d_h": df["Dest"].astype(str) + "_" + h.astype(str),
        "c_h": df["UniqueCarrier"].astype(str) + "_" + h.astype(str),
        "r": route,
        "r_h": route + "_" + h.astype(str),
    }


_cnt_maps = {k: v.value_counts() for k, v in _cnt_keys(train).items()}
_cnt_min = {k: float(v.min()) for k, v in _cnt_maps.items()}

# Smoothed target encoding on the same interaction keys; fit on train only (OOF for training matrix).
TE_KEYS = ["o_h", "r", "r_h"]
TE_ALPHA = 30.0
_y = (train[TARGET] == POSITIVE).astype(int)
_prior = float(_y.mean())
_train_keys = _cnt_keys(train)
_te_maps = {}
for _k in TE_KEYS:
    _g = _y.groupby(_train_keys[_k]).agg(["sum", "count"])
    _te_maps[_k] = (_g["sum"] + _prior * TE_ALPHA) / (_g["count"] + TE_ALPHA)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CYCLIC:
        v = _cint(df[c])
        X[c] = v
        period = CYCLIC[c]
        X[f"{c}_sin"] = np.sin(2 * np.pi * v / period)
        X[f"{c}_cos"] = np.cos(2 * np.pi * v / period)
    # DepTime is hhmm; values >2400 are after midnight.
    dep = df["DepTime"].astype(float)
    hour = (dep // 100).astype(int) % 24
    minute = (dep % 100).astype(int)
    tod = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_tod"] = tod
    X["dep_tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    keys = _cnt_keys(df)
    for k, v in keys.items():
        X[f"{k}_cnt"] = v.map(_cnt_maps[k]).fillna(_cnt_min[k]).astype(float)
    for k in TE_KEYS:
        X[f"{k}_te"] = keys[k].map(_te_maps[k]).fillna(_prior).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_BAG = 60
DEPTHS = [4, 5, 6, 7, 8]
X_train = prepare(train)
y_train = to_y(train)

from sklearn.model_selection import KFold

_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for _k in TE_KEYS:
    _key = _train_keys[_k]
    _oof = np.full(len(train), _prior)
    for _tr, _va in _kf.split(train):
        _g = _y.iloc[_tr].groupby(_key.iloc[_tr]).agg(["sum", "count"])
        _m = (_g["sum"] + _prior * TE_ALPHA) / (_g["count"] + TE_ALPHA)
        _oof[_va] = _key.iloc[_va].map(_m).fillna(_prior).to_numpy()
    X_train[f"{_k}_te"] = _oof

t0 = time.time()
models = []
for i in range(N_BAG):
    m = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=DEPTHS[i % len(DEPTHS)],
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        min_child_weight=5.0,
        gamma=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
