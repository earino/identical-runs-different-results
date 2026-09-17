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

# congestion proxies: how many scheduled departures share the same airport/hour (and carrier/day) in train
_h_tr = ((train["DepTime"] // 100) % 24).astype(str)
_dow_tr = train["DayOfWeek"].astype(str)
FREQ = {
    "origin_hour": (train["Origin"].astype(str) + "_" + _h_tr).value_counts(),
    "dest_hour": (train["Dest"].astype(str) + "_" + _h_tr).value_counts(),
    "carrier_dow": (train["UniqueCarrier"].astype(str) + "_" + _dow_tr).value_counts(),
    "route": (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts(),
}


def _num(s: pd.Series) -> pd.Series:
    # "c-4" -> 4 ; plain numeric -> itself
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    dep = _num(X["DepTime"])
    hour = (dep // 100) % 24
    X["dep_hour"] = hour
    X["dep_minutes"] = hour * 60 + (dep % 100)
    X["month_num"] = _num(X["Month"])
    dow = _num(X["DayOfWeek"])
    X["dow_num"] = dow
    X["is_weekend"] = (dow >= 6).astype(int)

    orig = X["Origin"].astype(str)
    dest = X["Dest"].astype(str)
    hs = hour.astype(str)
    X["origin_hour"] = (orig + "_" + hs).map(FREQ["origin_hour"]).fillna(0).astype(float)
    X["dest_hour"] = (dest + "_" + hs).map(FREQ["dest_hour"]).fillna(0).astype(float)
    X["carrier_dow"] = (X["UniqueCarrier"].astype(str) + "_" + dow.astype(str)).map(FREQ["carrier_dow"]).fillna(0).astype(float)
    X["route_freq"] = (orig + "_" + dest).map(FREQ["route"]).fillna(0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.02,
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
