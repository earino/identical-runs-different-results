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
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# Month / DayofMonth / DayOfWeek are ordinal (`c-<n>` strings): use them numerically.
# The three true high-cardinality categoricals stay categorical for xgboost.
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# Traffic/popularity statistics fitted on TRAIN ONLY: schedules are stable across years,
# so route/airport frequency should transfer better than raw identity.
_tr_hour = ((pd.to_numeric(train["DepTime"], errors="coerce") // 100) % 24).astype("Int64")


def _counts(key: pd.Series) -> pd.Series:
    return key.value_counts()


FREQ = {
    "origin_cnt": _counts(train["Origin"]),
    "dest_cnt": _counts(train["Dest"]),
    "carrier_cnt": _counts(train["UniqueCarrier"]),
    "route_cnt": _counts(train["Origin"].astype(str) + "_" + train["Dest"].astype(str)),
    "origin_hour_cnt": _counts(train["Origin"].astype(str) + "_" + _tr_hour.astype(str)),
    "dest_hour_cnt": _counts(train["Dest"].astype(str) + "_" + _tr_hour.astype(str)),
}


def _cnum(s: pd.Series) -> pd.Series:
    """`c-12` -> 12 (numeric)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    month = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = ((dep // 100) % 24).astype("float64")
    minute = (dep % 100).astype("float64")
    tod = hour * 60 + minute                     # minute of day
    doy = (month - 1) * 31 + dom                 # day of year (approx.)

    X = pd.DataFrame(index=df.index)
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["doy"] = doy
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    h = hour.astype("Int64").astype(str)
    keys = {
        "origin_cnt": df["Origin"].astype(str),
        "dest_cnt": df["Dest"].astype(str),
        "carrier_cnt": df["UniqueCarrier"].astype(str),
        "route_cnt": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "origin_hour_cnt": df["Origin"].astype(str) + "_" + h,
        "dest_hour_cnt": df["Dest"].astype(str) + "_" + h,
    }
    for name, k in keys.items():
        X[name] = k.map(FREQ[name]).fillna(0).astype("float64")
    # Normalised congestion: how peaky is this airport/route at this hour (stable schedule shape).
    X["origin_hour_share"] = X["origin_hour_cnt"] / X["origin_cnt"].replace(0, np.nan)
    X["dest_hour_share"] = X["dest_hour_cnt"] / X["dest_cnt"].replace(0, np.nan)
    X["route_share"] = X["route_cnt"] / X["origin_cnt"].replace(0, np.nan)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Bagged ensemble of XGBoost models: averaging over seeds + row/column subsampling cuts
# the variance that a single fit picks up from year-specific noise.
SEEDS = [1, 2, 3, 4, 5]
MEMBER_CFGS = [
    dict(max_depth=5, colsample_bytree=0.6, subsample=0.7),
    dict(max_depth=5, colsample_bytree=1.0, subsample=0.9),
    dict(max_depth=6, colsample_bytree=0.6, subsample=0.9),
    dict(max_depth=6, colsample_bytree=0.8, subsample=0.8),
    dict(max_depth=6, colsample_bytree=1.0, subsample=0.7),
    dict(max_depth=7, colsample_bytree=0.6, subsample=0.8),
    dict(max_depth=7, colsample_bytree=1.0, subsample=0.9),
    dict(max_depth=8, colsample_bytree=0.8, subsample=0.7),
]
models = [
    xgb.XGBClassifier(
        n_estimators=120,
        learning_rate=0.05,
        min_child_weight=50,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
        **cfg,
    )
    for s, cfg in zip(SEEDS + SEEDS, MEMBER_CFGS)
]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
for m in models:
    m.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
