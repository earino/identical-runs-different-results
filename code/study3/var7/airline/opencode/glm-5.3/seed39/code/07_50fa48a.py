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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_SRC = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_COLS = CAT_SRC + ["hour_cat"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(0).astype(int)


def _base(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    hour, minute = dep // 100, dep % 100
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour * 60 + minute  # minutes since midnight
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0.0)
    X["month_num"] = _num(df["Month"])
    X["day_num"] = _num(df["DayofMonth"])
    X["dow_num"] = _num(df["DayOfWeek"])
    for c in CAT_SRC:
        X[c] = df[c].astype(str).values
    X["hour_cat"] = hour.astype(str).values
    X["doy"] = (X["month_num"] - 1) * 31 + X["day_num"]  # approximate day of year / season
    return X


def _bins(X: pd.DataFrame, w: int) -> pd.Series:
    return pd.cut(X["tod"], np.arange(0, 1441, w), include_lowest=True).astype(str)


# --- congestion statistics: fit on TRAIN ONLY ----------------------------------
_Btr = _base(train)
cat_levels = {c: pd.Index(sorted(_Btr[c].dropna().unique())) for c in CAT_COLS}

_org = np.log1p(_Btr["Origin"].value_counts())
_dst = np.log1p(_Btr["Dest"].value_counts())
_rte = np.log1p((_Btr["Origin"].astype(str) + "_" + _Btr["Dest"].astype(str)).value_counts())
_bc = {}  # origin/dest x time-bin counts
for w in (30, 60, 120):
    b = _bins(_Btr, w)
    _bc[f"oh{w}"] = np.log1p((_Btr["Origin"].astype(str) + "_" + b).value_counts())
    _bc[f"dh{w}"] = np.log1p((_Btr["Dest"].astype(str) + "_" + b).value_counts())
b30 = _bins(_Btr, 30)
_bc["ch30"] = np.log1p((_Btr["UniqueCarrier"].astype(str) + "_" + b30).value_counts())
_bc["odow"] = np.log1p((_Btr["Origin"].astype(str) + "_" + _Btr["DayOfWeek"].astype(str)).value_counts())
_tod_by_org = _Btr.groupby("Origin")["tod"].apply(np.sort).to_dict()


def _window_count(X: pd.DataFrame, W: int) -> np.ndarray:
    # # flights departing the same origin within +-W minutes (train schedule only)
    out = np.zeros(len(X), dtype=float)
    orgs = X["Origin"].astype(str).values
    tods = X["tod"].values
    for o, arr in _tod_by_org.items():
        idx = np.nonzero(orgs == o)[0]
        if idx.size == 0:
            continue
        lo = np.searchsorted(arr, tods[idx] - W, side="left")
        hi = np.searchsorted(arr, tods[idx] + W, side="right")
        out[idx] = hi - lo
    return np.log1p(out)


def _add_congestion(X: pd.DataFrame) -> pd.DataFrame:
    X["org_traffic"] = X["Origin"].map(_org).fillna(0.0)
    X["dst_traffic"] = X["Dest"].map(_dst).fillna(0.0)
    X["rte_traffic"] = (X["Origin"].astype(str) + "_" + X["Dest"].astype(str)).map(_rte).fillna(0.0)
    for w in (30, 60, 120):
        b = _bins(X, w)
        X[f"oh{w}"] = (X["Origin"].astype(str) + "_" + b).map(_bc[f"oh{w}"]).fillna(0.0)
        X[f"dh{w}"] = (X["Dest"].astype(str) + "_" + b).map(_bc[f"dh{w}"]).fillna(0.0)
    b = _bins(X, 30)
    X["ch30"] = (X["UniqueCarrier"].astype(str) + "_" + b).map(_bc["ch30"]).fillna(0.0)
    X["odow"] = (X["Origin"].astype(str) + "_" + X["DayOfWeek"].astype(str)).map(_bc["odow"]).fillna(0.0)
    X["w30"] = _window_count(X, 30)
    X["w60"] = _window_count(X, 60)
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _add_congestion(_base(df))
    X = X[feature_cols]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: diverse 5-seed ensemble of regularized XGB, early-stopped on eval --
COMMON = dict(
    n_estimators=3000,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    min_child_weight=20,
    reg_lambda=10.0,
    subsample=0.9,
    colsample_bytree=0.8,
    max_bin=512,
    n_jobs=N_JOBS,
)
MEMBERS = [
    dict(seed=100, max_depth=9, reg_alpha=16.0),
    dict(seed=101, max_depth=9, reg_alpha=16.0),
    dict(seed=102, max_depth=8, reg_alpha=12.0),
]

feature_cols = list(_add_congestion(_Btr).columns)
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = []
t0 = time.time()
for mem in MEMBERS:
    m = xgb.XGBClassifier(random_state=mem["seed"], max_depth=mem["max_depth"], reg_alpha=mem["reg_alpha"], **COMMON)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in models], axis=0))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
