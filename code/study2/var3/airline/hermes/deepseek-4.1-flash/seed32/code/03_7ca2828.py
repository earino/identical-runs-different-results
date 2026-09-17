"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives inside prepare(), which predict_proba() calls on unseen rows.
"""
import json
import os
import re
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

# --- feature configuration (derived from the TRAINING frame only) --------------
RAW_FEATURES = [c for c in train.columns if c not in ID_COLS + [TARGET]]

_C_TOKEN = re.compile(r"^c-(\d+)$")


def _is_c_token(series: pd.Series) -> bool:
    """True if every non-null value looks like 'c-<int>' (i.e. an ordinal coded as a string)."""
    vals = series.dropna().unique()
    return len(vals) > 0 and all(_C_TOKEN.match(str(v)) for v in vals)


# string columns that are really ordinal ints: Month / DayofMonth / DayOfWeek
ORD_COLS = [c for c in RAW_FEATURES if pd.api.types.is_string_dtype(train[c])
            or pd.api.types.is_object_dtype(train[c])]
ORD_COLS = [c for c in ORD_COLS if _is_c_token(train[c])]

# remaining string columns are true categoricals
CAT_COLS = [c for c in RAW_FEATURES
            if (pd.api.types.is_string_dtype(train[c]) or pd.api.types.is_object_dtype(train[c]))
            and c not in ORD_COLS and train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# DepTime (hhmm) -> hour / minute / minutes-of-day. Some encodings use 2400 for midnight.
DEPTIME = "DepTime" if "DepTime" in RAW_FEATURES else None
HAS_ROUTE = "Origin" in RAW_FEATURES and "Dest" in RAW_FEATURES

# log-frequency signals: log1p(count in training) for route / carrier / origin / dest
count_maps = {}
if HAS_ROUTE:
    _rt = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
    _vc = _rt.value_counts()
    count_maps = {k: {kk: float(np.log1p(vv)) for kk, vv in v.items()} for k, v in {
        "route": _vc,
        "uniquecarrier": train["UniqueCarrier"].astype(str).value_counts(),
        "origin": train["Origin"].astype(str).value_counts(),
        "dest": train["Dest"].astype(str).value_counts(),
    }.items()}


def _route_key(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _hhmm(df: pd.DataFrame) -> tuple:
    t = pd.to_numeric(df[DEPTIME], errors="coerce")
    t = t.where(t < 2400, t - 2400)
    hour = np.floor(t / 100.0)
    minute = t - hour * 100.0
    hour = hour.where((hour >= 0) & (hour <= 23), np.nan)
    minute = minute.where((minute >= 0) & (minute <= 59), np.nan)
    return hour, minute


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[RAW_FEATURES].copy()
    for c in ORD_COLS:
        X[c] = pd.to_numeric(X[c].astype(str).str.replace("^c-", "", regex=True), errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    if DEPTIME is not None:
        hour, minute = _hhmm(df)
        X["dep_hour"] = hour
        X["dep_minute"] = minute
        X["dep_minofday"] = hour * 60.0 + minute
    if HAS_ROUTE:
        X["route_cnt"] = _route_key(df).map(count_maps["route"]).fillna(0.0).to_numpy()
        for c in ("UniqueCarrier", "Origin", "Dest"):
            X[c.lower() + "_cnt"] = df[c].astype(str).map(count_maps[c.lower()]).fillna(0.0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Small ensemble of XGBoost models (seed averaging) at one strongly regularized
# lossguide configuration. Averaging reduces variance without fitting eval.
PARAMS = dict(
    n_estimators=500,
    learning_rate=0.02,
    grow_policy="lossguide",
    max_depth=0,
    max_leaves=256,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_lambda=20.0,
    max_bin=512,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
N_SEEDS = 3
X_TRAIN, Y_TRAIN = prepare(train), to_y(train)
MODELS = []
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(random_state=SEED + 100 * s, **PARAMS)
    t0 = time.time()
    m.fit(X_TRAIN, Y_TRAIN)
    print(f"Training time seed {s}: {time.time() - t0:.1f}s")
    MODELS.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
