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

# stable structural features: flight counts per group, fit on train only
_train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_train_hour = (train["DepTime"].astype(float) // 100).clip(0, 23).astype(int)
_train_undir = pd.Series(["_".join(sorted(p)) for p in zip(train["Origin"], train["Dest"])])
FREQ = {
    "origin_count": train["Origin"].value_counts(),
    "dest_count": train["Dest"].value_counts(),
    "carrier_count": train["UniqueCarrier"].value_counts(),
    "route_count": _train_route.value_counts(),
    "origin_hour_count": pd.Series(list(zip(train["Origin"], _train_hour))).value_counts(),
    "dest_hour_count": pd.Series(list(zip(train["Dest"], _train_hour))).value_counts(),
    "carrier_hour_count": pd.Series(list(zip(train["UniqueCarrier"], _train_hour))).value_counts(),
    "undir_route_count": _train_undir.value_counts(),
    "carrier_origin_count": pd.Series(list(zip(train["UniqueCarrier"], train["Origin"]))).value_counts(),
}


def _num(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dep = X["DepTime"].astype(float)
    hour = (dep // 100).clip(0, 23).astype(int)
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = hour + X["minute"] / 60.0
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["origin_count"] = df["Origin"].map(FREQ["origin_count"]).fillna(0).astype(float)
    X["dest_count"] = df["Dest"].map(FREQ["dest_count"]).fillna(0).astype(float)
    X["carrier_count"] = df["UniqueCarrier"].map(FREQ["carrier_count"]).fillna(0).astype(float)
    X["route_count"] = route.map(FREQ["route_count"]).fillna(0).astype(float)
    X["origin_hour_count"] = pd.Series(list(zip(df["Origin"], hour))).map(FREQ["origin_hour_count"]).fillna(0).to_numpy(dtype=float)
    X["dest_hour_count"] = pd.Series(list(zip(df["Dest"], hour))).map(FREQ["dest_hour_count"]).fillna(0).to_numpy(dtype=float)
    X["carrier_hour_count"] = pd.Series(list(zip(df["UniqueCarrier"], hour))).map(FREQ["carrier_hour_count"]).fillna(0).to_numpy(dtype=float)
    undir = pd.Series(["_".join(sorted(p)) for p in zip(df["Origin"], df["Dest"])])
    X["undir_route_count"] = undir.map(FREQ["undir_route_count"]).fillna(0).to_numpy(dtype=float)
    X["carrier_origin_count"] = pd.Series(list(zip(df["UniqueCarrier"], df["Origin"]))).map(FREQ["carrier_origin_count"]).fillna(0).to_numpy(dtype=float)
    # relative congestion / share features
    X["origin_hour_share"] = X["origin_hour_count"] / X["origin_count"].clip(lower=1)
    X["route_share"] = X["route_count"] / X["origin_count"].clip(lower=1)
    X["carrier_origin_share"] = X["carrier_origin_count"] / X["origin_count"].clip(lower=1)
    X["undir_route_share"] = X["undir_route_count"] / X["origin_count"].clip(lower=1)
    X["dest_hour_share"] = X["dest_hour_count"] / X["dest_count"].clip(lower=1)
    X["carrier_hour_share"] = X["carrier_hour_count"] / X["carrier_count"].clip(lower=1)
    # holiday / seasonality flags (identifiable without year via day-of-week)
    mon = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    thanksgiving = (mon == 11) & (dow == 4) & dom.between(22, 28)
    labor = (mon == 9) & (dow == 1) & (dom <= 7)
    memorial = (mon == 5) & (dow == 1) & (dom >= 25)
    fixed = ((mon == 1) & (dom <= 2)) | ((mon == 7) & dom.between(3, 5)) | ((mon == 12) & dom.between(24, 26))
    X["is_holiday"] = (thanksgiving | labor | memorial | fixed).astype(float).to_numpy()
    window = fixed | thanksgiving | ((mon == 12) & (dom >= 18)) | ((mon == 1) & (dom <= 4)) | ((mon == 11) & dom.between(21, 29))
    X["is_holiday_window"] = window.astype(float).to_numpy()
    X["is_weekend"] = (dow >= 6).astype(float).to_numpy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=0,
    grow_policy="lossguide",
    max_leaves=256,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.4,
    reg_lambda=2.0,
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
