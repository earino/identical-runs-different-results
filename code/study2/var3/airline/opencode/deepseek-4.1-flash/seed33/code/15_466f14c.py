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
def _hour_bucket(t):
    hour = np.floor(np.asarray(t, dtype=float) / 100.0)
    hour = np.where((hour < 0) | (hour > 23), -1, hour)
    return np.floor(hour / 2.0).astype(int)


# frequency statistics fitted on training data only
_train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_train_hb = pd.Series(_hour_bucket(train["DepTime"].to_numpy()), index=train.index).astype(str)
_train_origin = train["Origin"].astype(str)
_train_dest = train["Dest"].astype(str)
_train_carrier = train["UniqueCarrier"].astype(str)
_train_dow = train["DayOfWeek"].astype(str)
_freq_train = {
    "Origin": _train_origin,
    "Dest": _train_dest,
    "UniqueCarrier": _train_carrier,
    "Route": _train_route,
    "Origin_hour": _train_origin + "_" + _train_hb,
    "Dest_hour": _train_dest + "_" + _train_hb,
    "Carrier_hour": _train_carrier + "_" + _train_hb,
    "Origin_dow": _train_origin + "_" + _train_dow,
    "Dest_dow": _train_dest + "_" + _train_dow,
    "Carrier_dow": _train_carrier + "_" + _train_dow,
}
freq_maps = {k: v.value_counts(normalize=True) for k, v in _freq_train.items()}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    t = df["DepTime"].astype(float).to_numpy()
    hour = np.floor(t / 100.0)
    hour = np.where((hour < 0) | (hour > 23), np.nan, hour)
    minute = np.where(np.isnan(hour), np.nan, t - hour * 100.0)
    dep_clock = hour * 60.0 + minute
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_clock"] = dep_clock
    X["dep_sin"] = np.sin(2 * np.pi * dep_clock / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dep_clock / 1440.0)

    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    route = origin + "_" + dest
    hb = pd.Series(_hour_bucket(df["DepTime"].to_numpy()), index=df.index).astype(str)
    dow = df["DayOfWeek"].astype(str)
    freq_keys = {
        "Origin": origin,
        "Dest": dest,
        "UniqueCarrier": carrier,
        "Route": route,
        "Origin_hour": origin + "_" + hb,
        "Dest_hour": dest + "_" + hb,
        "Carrier_hour": carrier + "_" + hb,
        "Origin_dow": origin + "_" + dow,
        "Dest_dow": dest + "_" + dow,
        "Carrier_dow": carrier + "_" + dow,
    }
    for k, ser in freq_keys.items():
        X[k + "_freq"] = ser.map(freq_maps[k]).astype(float).to_numpy()
    X["origin_hour_share"] = (X["Origin_hour_freq"] / X["Origin_freq"]).astype(float)
    X["dest_hour_share"] = (X["Dest_hour_freq"] / X["Dest_freq"]).astype(float)
    X["carrier_hour_share"] = (X["Carrier_hour_freq"] / X["UniqueCarrier_freq"]).astype(float)
    X["route_share"] = (X["Route_freq"] / (X["Origin_freq"] * X["Dest_freq"])).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=8000,
    max_depth=24,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.3,
    min_child_weight=1,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=80,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
