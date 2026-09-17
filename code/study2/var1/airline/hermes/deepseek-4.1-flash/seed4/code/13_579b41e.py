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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

RAW_CATS = ["Month", "DayofMonth", "DayOfWeek"]


def _ordinal(s: pd.Series) -> pd.Series:
    """'c-12' -> 12.0 for the c-<n> encoded columns."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").astype("float32")


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype("int64") // 100).clip(0, 24)


def _tod(df: pd.DataFrame) -> pd.Series:
    return (_hour(df) * 60 + (df["DepTime"].astype("int64") % 100).clip(0, 59)).astype("int64")


def _b30(df: pd.DataFrame) -> pd.Series:
    return (_tod(df) // 30).astype(str)


def _b15(df: pd.DataFrame) -> pd.Series:
    return (_tod(df) // 15).astype(str)


def _b10(df: pd.DataFrame) -> pd.Series:
    return (_tod(df) // 10).astype(str)


def _org(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str)


def _dst(df: pd.DataFrame) -> pd.Series:
    return df["Dest"].astype(str)


def _car(df: pd.DataFrame) -> pd.Series:
    return df["UniqueCarrier"].astype(str)


# target-encoding keys: name -> (key series builder, smoothing strength k)
KEYS = {
    "carrier": (_car, 100.0),
    "origin": (_org, 100.0),
    "dest": (_dst, 100.0),
    "route": (lambda d: _org(d) + "_" + _dst(d), 50.0),
    "hour": (lambda d: _hour(d).astype(str), 100.0),
    "car_hour": (lambda d: _car(d) + "_" + _hour(d).astype(str), 100.0),
    "org_hour": (lambda d: _org(d) + "_" + _hour(d).astype(str), 50.0),
    "dest_hour": (lambda d: _dst(d) + "_" + _hour(d).astype(str), 50.0),
    "car_dow": (lambda d: _car(d) + "_" + _ordinal(d["DayOfWeek"]).astype(str), 100.0),
    "org_dow": (lambda d: _org(d) + "_" + _ordinal(d["DayOfWeek"]).astype(str), 50.0),
    "car_dest": (lambda d: _car(d) + "_" + _dst(d), 50.0),
    "car_org": (lambda d: _car(d) + "_" + _org(d), 50.0),
    "route_hour": (lambda d: _org(d) + "_" + _dst(d) + "_" + _hour(d).astype(str), 30.0),
    "route_dow": (lambda d: _org(d) + "_" + _dst(d) + "_" + _ordinal(d["DayOfWeek"]).astype(str), 30.0),
    "org_hour_dow": (lambda d: _org(d) + "_" + _hour(d).astype(str) + "_" + _ordinal(d["DayOfWeek"]).astype(str), 30.0),
    "car_route_hour": (lambda d: _car(d) + "_" + _org(d) + "_" + _dst(d) + "_" + _hour(d).astype(str), 20.0),
    "route_b30": (lambda d: _org(d) + "_" + _dst(d) + "_" + _b30(d), 20.0),
    "org_b30": (lambda d: _org(d) + "_" + _b30(d), 30.0),
    "car_b30": (lambda d: _car(d) + "_" + _b30(d), 30.0),
    "route_b15": (lambda d: _org(d) + "_" + _dst(d) + "_" + _b15(d), 15.0),
    "org_b15": (lambda d: _org(d) + "_" + _b15(d), 20.0),
    "dest_b30": (lambda d: _dst(d) + "_" + _b30(d), 30.0),
    "route_b10": (lambda d: _org(d) + "_" + _dst(d) + "_" + _b10(d), 10.0),
    "org_b10": (lambda d: _org(d) + "_" + _b10(d), 15.0),
    "dest_b15": (lambda d: _dst(d) + "_" + _b15(d), 20.0),
}

FREQ = {
    "carrier": _car,
    "origin": _org,
    "dest": _dst,
    "route": lambda d: _org(d) + "_" + _dst(d),
    "org_hour": lambda d: _org(d) + "_" + _hour(d).astype(str),
    "dest_hour": lambda d: _dst(d) + "_" + _hour(d).astype(str),
    "car_org": lambda d: _car(d) + "_" + _org(d),
    "route_hour": lambda d: _org(d) + "_" + _dst(d) + "_" + _hour(d).astype(str),
    "route_dow": lambda d: _org(d) + "_" + _dst(d) + "_" + _ordinal(d["DayOfWeek"]).astype(str),
    "org_hour_dow": lambda d: _org(d) + "_" + _hour(d).astype(str) + "_" + _ordinal(d["DayOfWeek"]).astype(str),
    "car_route_hour": lambda d: _car(d) + "_" + _org(d) + "_" + _dst(d) + "_" + _hour(d).astype(str),
    "route_b30": lambda d: _org(d) + "_" + _dst(d) + "_" + _b30(d),
    "org_b30": lambda d: _org(d) + "_" + _b30(d),
    "car_b30": lambda d: _car(d) + "_" + _b30(d),
    "route_b15": lambda d: _org(d) + "_" + _dst(d) + "_" + _b15(d),
    "org_b15": lambda d: _org(d) + "_" + _b15(d),
    "dest_b30": lambda d: _dst(d) + "_" + _b30(d),
    "route_b10": lambda d: _org(d) + "_" + _dst(d) + "_" + _b10(d),
    "org_b10": lambda d: _org(d) + "_" + _b10(d),
    "dest_b15": lambda d: _dst(d) + "_" + _b15(d),
}
FREQ_COLS = [f"n_{k}" for k in FREQ]
FREQ_MAP = {}
for _name, _fn in FREQ.items():
    _v = _fn(train)
    FREQ_MAP[_name] = np.log1p(_v.groupby(_v).size())

y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_train.mean())
PRIOR_LOGIT = float(np.log(PRIOR / (1 - PRIOR)))
TE_COLS = [f"te_{k}" for k in KEYS]

# full-train statistics: used for eval / hidden holdout
TE_MAP = {}
for _name, (_fn, _k) in KEYS.items():
    g = pd.DataFrame({"k": _fn(train).to_numpy(), "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    p = (g["sum"] + PRIOR * _k) / (g["count"] + _k)
    TE_MAP[_name] = np.log(p / (1 - p))

# out-of-fold statistics: used for the training rows themselves (avoids leakage)
OOF = pd.DataFrame(index=train.index, columns=TE_COLS, dtype="float64")
for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
    _y = y_train[_tr]
    for _name, (_fn, _k) in KEYS.items():
        _ktr = _fn(train.iloc[_tr])
        _kva = _fn(train.iloc[_va])
        g = pd.DataFrame({"k": _ktr.to_numpy(), "y": _y}).groupby("k")["y"].agg(["sum", "count"])
        p = (g["sum"] + PRIOR * _k) / (g["count"] + _k)
        OOF.loc[train.index[_va], f"te_{_name}"] = _kva.map(np.log(p / (1 - p))).to_numpy()
OOF = OOF.fillna(PRIOR_LOGIT).astype("float32")

feature_cols = [
    "DepTime", "tod", "hour", "minute", "tod_sin", "tod_cos",
    "Distance", "log_dist",
    "month_i", "dom_i", "dow_i",
] + RAW_CATS + TE_COLS + FREQ_COLS

cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CATS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    dep = df["DepTime"].astype("float32")
    hour = (dep // 100).clip(0, 24)
    minute = (dep % 100).clip(0, 59)
    tod = hour * 60 + minute
    dist = df["Distance"].astype("float32")
    m = _ordinal(df["Month"])
    d = _ordinal(df["DayofMonth"])
    w = _ordinal(df["DayOfWeek"])

    X = pd.DataFrame(index=df.index)
    X["DepTime"] = dep
    X["tod"] = tod
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["month_i"] = m
    X["dom_i"] = d
    X["dow_i"] = w
    for c in RAW_CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])

    oof = df is train
    for name, (fn, _k) in KEYS.items():
        if oof:
            X[f"te_{name}"] = OOF[f"te_{name}"].to_numpy()
        else:
            X[f"te_{name}"] = fn(df).map(TE_MAP[name]).fillna(PRIOR_LOGIT).to_numpy()
    for name, fn in FREQ.items():
        X[f"n_{name}"] = fn(df).map(FREQ_MAP[name]).fillna(0.0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=40,
    reg_lambda=10.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
