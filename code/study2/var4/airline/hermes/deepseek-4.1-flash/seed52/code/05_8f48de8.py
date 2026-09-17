"""XGBoost binary classifier for the airline task (see program.md).

Contract:
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
# Baseline kept Month/DayOfWeek as c-<n> strings and DepTime as a raw hhmm integer.
# Both are poor encodings for a tree model: hhmm wraps around (2359 -> 0000) and the
# string levels carry no order. Decompose DepTime into hour/minute/time-of-day and turn
# the c-<n> columns into real numbers.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
# category levels learned on TRAIN only; unseen levels at predict time become NaN.
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _carrier_hour(df: pd.DataFrame) -> pd.Series:
    return df["UniqueCarrier"].astype(str) + "_" + df["DepTime"].astype(str).str.slice(0, 2)


def _hour_of(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23).astype("Int64").astype(str)


# Target-encoded keys: high-cardinality / interaction features that XGBoost cannot split on
# directly without memorising levels. Each is summarised by a smoothed target mean.
TE_SPECS = {
    "route": _route,
    "carrier": lambda df: df["UniqueCarrier"].astype(str),
    "origin": lambda df: df["Origin"].astype(str),
    "dest": lambda df: df["Dest"].astype(str),
    "origin_hour": lambda df: df["Origin"].astype(str) + "_" + _hour_of(df),
    "dest_hour": lambda df: df["Dest"].astype(str) + "_" + _hour_of(df),
    "carrier_hour": _carrier_hour,
    "route_hour": lambda df: _route(df) + "_" + _hour_of(df),
    "carrier_route": lambda df: df["UniqueCarrier"].astype(str) + "_" + _route(df),
    "origin_month": lambda df: df["Origin"].astype(str) + "_" + df["Month"].astype(str),
    "dest_month": lambda df: df["Dest"].astype(str) + "_" + df["Month"].astype(str),
    "carrier_month": lambda df: df["UniqueCarrier"].astype(str) + "_" + df["Month"].astype(str),
    "origin_dow": lambda df: df["Origin"].astype(str) + "_" + df["DayOfWeek"].astype(str),
    "dest_dow": lambda df: df["Dest"].astype(str) + "_" + df["DayOfWeek"].astype(str),
}
TE_K = 20.0
PRIOR = float((train[TARGET] == POSITIVE).mean())
FOLDS = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(len(train))))


def _te_lookup(key: pd.Series, y: np.ndarray) -> pd.Series:
    st = pd.DataFrame({"k": np.asarray(key), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (st["sum"] + PRIOR * TE_K) / (st["count"] + TE_K)


# full-train lookups used at predict time
TE_LUT = {name: _te_lookup(fn(train), (train[TARGET] == POSITIVE).astype(float).to_numpy()) for name, fn in TE_SPECS.items()}
# out-of-fold values the model is fitted on, so training rows never see their own target
TE_OOF = {}
for _name, _fn in TE_SPECS.items():
    _key = _fn(train)
    _y = (train[TARGET] == POSITIVE).astype(float).to_numpy()
    _oof = np.full(len(train), PRIOR)
    for _tr, _va in FOLDS:
        _lut = _te_lookup(_key.iloc[_tr], _y[_tr])
        _oof[_va] = _key.iloc[_va].map(_lut).fillna(PRIOR).to_numpy()
    TE_OOF[_name] = _oof


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    dow = _cnum(df["DayOfWeek"])
    day = _cnum(df["DayofMonth"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    minute = dep % 100
    tod = hour * 60 + minute

    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["is_weekend"] = (dow >= 6).astype(int)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    for name, fn in TE_SPECS.items():
        X[f"te_{name}"] = fn(df).map(TE_LUT[name]).fillna(PRIOR).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model = xgb.XGBClassifier(**PARAMS)

t0 = time.time()
X_train = prepare(train)
for name in TE_SPECS:
    X_train[f"te_{name}"] = TE_OOF[name]  # out-of-fold values for the fitted rows
model.fit(X_train, to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
