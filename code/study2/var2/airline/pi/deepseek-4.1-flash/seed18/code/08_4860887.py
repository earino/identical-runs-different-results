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
_FREQ = {
    "origin_freq": train["Origin"].value_counts().to_dict(),
    "dest_freq": train["Dest"].value_counts().to_dict(),
    "route_freq": (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts().to_dict(),
    "carrier_freq": train["UniqueCarrier"].value_counts().to_dict(),
}


def _key_pair(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(str) + "_" + b.astype(str)


_FREQ["origin_carrier_freq"] = _key_pair(train["Origin"], train["UniqueCarrier"]).value_counts().to_dict()
_FREQ["dest_carrier_freq"] = _key_pair(train["Dest"], train["UniqueCarrier"]).value_counts().to_dict()
_FREQ["route_carrier_freq"] = _key_pair(_key_pair(train["Origin"], train["Dest"]), train["UniqueCarrier"]).value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["origin_freq"] = df["Origin"].map(_FREQ["origin_freq"]).fillna(0).astype(float)
    X["dest_freq"] = df["Dest"].map(_FREQ["dest_freq"]).fillna(0).astype(float)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_freq"] = route.map(_FREQ["route_freq"]).fillna(0).astype(float)
    X["carrier_freq"] = df["UniqueCarrier"].map(_FREQ["carrier_freq"]).fillna(0).astype(float)
    X["origin_carrier_freq"] = _key_pair(df["Origin"], df["UniqueCarrier"]).map(_FREQ["origin_carrier_freq"]).fillna(0).astype(float)
    X["dest_carrier_freq"] = _key_pair(df["Dest"], df["UniqueCarrier"]).map(_FREQ["dest_carrier_freq"]).fillna(0).astype(float)
    X["route_carrier_freq"] = _key_pair(route, df["UniqueCarrier"]).map(_FREQ["route_carrier_freq"]).fillna(0).astype(float)
    month = df["Month"].str.slice(2).astype(int)
    day = df["DayofMonth"].str.slice(2).astype(int)
    doy = pd.to_datetime(dict(year=2005, month=month, day=day)).dt.dayofyear.astype(float)
    X["day_of_year"] = doy
    # proximity to major US travel holidays (approximate day-of-year, circular)
    hol = np.array([1.0, 148.0, 185.0, 246.0, 327.0, 359.0])
    diff = np.abs(doy.to_numpy()[:, None] - hol[None, :])
    diff = np.minimum(diff, 365.0 - diff)
    X["holiday_dist"] = diff.min(axis=1)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
MODEL_SPECS = [
    dict(n_estimators=200, max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42),
    dict(n_estimators=250, max_depth=4, learning_rate=0.05, min_child_weight=8, subsample=0.7, colsample_bytree=0.6, random_state=7),
    dict(n_estimators=300, max_depth=5, learning_rate=0.04, min_child_weight=15, subsample=0.8, colsample_bytree=0.7, random_state=123),
    dict(n_estimators=400, max_depth=3, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=2024),
    dict(n_estimators=150, max_depth=6, learning_rate=0.06, min_child_weight=20, subsample=0.7, colsample_bytree=0.5, random_state=555),
    dict(n_estimators=250, max_depth=8, learning_rate=0.04, min_child_weight=30, subsample=0.8, colsample_bytree=0.7, random_state=888, grow_policy="lossguide", max_leaves=32),
    dict(n_estimators=350, max_depth=4, learning_rate=0.04, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=13),
    dict(n_estimators=220, max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.9, colsample_bytree=0.7, random_state=99),
]

X_train = prepare(train)
y_train = to_y(train)
models = []
t0 = time.time()
for spec in MODEL_SPECS:
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **spec,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
