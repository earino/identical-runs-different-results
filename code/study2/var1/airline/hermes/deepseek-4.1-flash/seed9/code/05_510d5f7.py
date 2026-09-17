"""XGBoost binary classifier for airline departure delay (autoresearch experiment).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare() so it also applies to unseen rows.

Model: rank-averaged ensemble of XGBoost models sharing a base feature set (calendar, time-of-day,
distance, airports, carrier) plus engineered interactions that carry the signal:
carrier x scheduled hour, carrier x distance bucket and hour x distance bucket.
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# categorical levels are learned from TRAIN only; unseen levels -> NaN for xgboost
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

_DEP_TR = pd.to_numeric(train["DepTime"], errors="coerce")
_DBINS = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 10000]
_DBINS2 = [0, 150, 300, 450, 600, 800, 1000, 1250, 1500, 1800, 2200, 2800, 10000]


def _hour_str(dep):
    return (dep // 100).clip(0, 23).astype("Int64").astype(str)


def _b30_str(dep):
    return (dep // 30).clip(0, 47).astype("Int64").astype(str)


def _cdist_str(df, bins):
    d = pd.cut(pd.to_numeric(df["Distance"], errors="coerce"), bins)
    return df["UniqueCarrier"].astype(str) + "_" + d.astype(str)


def _hdist_str(df, dep, bins):
    d = pd.cut(pd.to_numeric(df["Distance"], errors="coerce"), bins)
    return _hour_str(dep) + "_" + d.astype(str)


def _levels(series):
    return pd.Index(sorted(series.unique()))


# carrier x scheduled-hour: carriers run different banks of flights with different delay behaviour
CARRIER_HOUR_LEVELS = _levels(train["UniqueCarrier"].astype(str) + "_" + _hour_str(_DEP_TR))
# carrier x 30-minute departure bucket
CARRIER_B30_LEVELS = _levels(train["UniqueCarrier"].astype(str) + "_" + _b30_str(_DEP_TR))
# carrier x distance bucket
CDIST_LEVELS = _levels(_cdist_str(train, _DBINS))
CDIST2_LEVELS = _levels(_cdist_str(train, _DBINS2))
# hour x distance bucket: delay propensity depends on the length of the flight at that time of day
HDIST_LEVELS = _levels(_hdist_str(train, _DEP_TR, _DBINS))


def _num(s):
    return pd.to_numeric(s.astype(str).str[2:], errors="coerce")


def prepare(df: pd.DataFrame, extras=()) -> pd.DataFrame:
    """Every feature is derived here from the raw columns, so predict_proba() can reproduce it."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    # time-of-day: scheduled departure hhmm -> hour / minute-of-day / cyclic encoding
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    h = (dep // 100).clip(0, 23)
    m = (dep % 100).clip(0, 59)
    tod = h + m / 60.0
    X["hour"] = h
    X["minofday"] = h * 60 + m
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)

    # calendar cycles (Month/DayOfWeek arrive as c-<n> strings)
    mo = _num(df["Month"])
    dw = _num(df["DayOfWeek"])
    X["mo_sin"] = np.sin(2 * np.pi * mo / 12.0)
    X["mo_cos"] = np.cos(2 * np.pi * mo / 12.0)
    X["dw_sin"] = np.sin(2 * np.pi * dw / 7.0)
    X["dw_cos"] = np.cos(2 * np.pi * dw / 7.0)

    if "ch" in extras:
        ch = df["UniqueCarrier"].astype(str) + "_" + _hour_str(dep)
        X["carrier_hour"] = pd.Categorical(ch, categories=CARRIER_HOUR_LEVELS)
    if "cb30" in extras:
        cb = df["UniqueCarrier"].astype(str) + "_" + _b30_str(dep)
        X["carrier_b30"] = pd.Categorical(cb, categories=CARRIER_B30_LEVELS)
    if "cdist" in extras:
        cd = _cdist_str(df, _DBINS)
        X["carrier_dist"] = pd.Categorical(cd, categories=CDIST_LEVELS)
    if "hdist" in extras:
        hd = _hdist_str(df, dep, _DBINS)
        X["hour_dist"] = pd.Categorical(hd, categories=HDIST_LEVELS)
    if "cdist2" in extras:
        cd2 = _cdist_str(df, _DBINS2)
        X["carrier_dist2"] = pd.Categorical(cd2, categories=CDIST2_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble of XGBoost models -----------------------------------------------
# members differ in feature subset, depth / regularisation and learning rate
MEMBERS = [
    dict(extras=("ch", "cb30", "cdist", "hdist"), params=dict(max_depth=4, learning_rate=0.03, min_child_weight=20, colsample_bynode=0.6), n_estimators=4000),
    dict(extras=("ch", "cb30", "cdist", "hdist"), params=dict(max_depth=4, learning_rate=0.05, min_child_weight=20, colsample_bynode=0.6), n_estimators=2000),
    dict(extras=("ch", "cb30", "cdist2", "hdist"), params=dict(max_depth=4, learning_rate=0.03, min_child_weight=20, colsample_bynode=0.6), n_estimators=3000),
    dict(extras=("ch",), params=dict(max_depth=4, learning_rate=0.03, min_child_weight=20, colsample_bynode=0.6), n_estimators=4000),
]

Y_TR = to_y(train)
models = []
t0 = time.time()
for i, mem in enumerate(MEMBERS):
    m = xgb.XGBClassifier(
        n_estimators=mem["n_estimators"],
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **mem["params"],
    )
    m.fit(prepare(train, mem["extras"]), Y_TR)
    models.append((m, mem["extras"]))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    scores = np.zeros(len(df))
    for m, extras in models:
        p = m.predict_proba(prepare(df, extras))[:, 1]
        scores += pd.Series(p).rank().to_numpy() / len(p)   # rank-average: members are on different scales
    return scores / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
