"""XGBoost binary classifier for airline departure delay (dep_delayed_15min).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  All feature engineering lives inside prepare(), which predict_proba() applies to unseen rows too.
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

# --- feature schema (derived from TRAIN only; never from the df passed in) -----
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
NUM_COLS = [c for c in feature_cols if c not in cat_cols]
TIME_COLS = ["hour", "minute", "tod", "tod_sin", "tod_cos", "is_weekend", "is_red_eye", "log_dist",
             "origin_freq", "dest_freq", "route_freq", "carrier_freq"]
# train-only frequency maps for airports / routes / carriers (no target information)
_route_train = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str))
_freq_maps = {
    "origin_freq": train["Origin"].value_counts(normalize=True).to_dict(),
    "dest_freq": train["Dest"].value_counts(normalize=True).to_dict(),
    "route_freq": _route_train.value_counts(normalize=True).to_dict(),
    "carrier_freq": train["UniqueCarrier"].value_counts(normalize=True).to_dict(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe (target may be absent) -> model matrix. Stateless: uses only train-derived maps."""
    X = df[NUM_COLS].copy()

    # scheduled departure time (hhmm int; 2400 means midnight)
    dt = df["DepTime"].to_numpy(dtype=np.int64) % 2400
    hour = dt // 100
    minute = dt % 100
    tod = (hour * 60 + minute).astype(np.float64)
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["is_red_eye"] = ((tod < 6 * 60) | (tod >= 22 * 60)).astype(np.int8)
    X["log_dist"] = np.log1p(df["Distance"].to_numpy(dtype=np.float64))

    dow = df["DayOfWeek"].str.slice(2).astype(int).to_numpy()
    X["is_weekend"] = (dow >= 6).astype(np.int8)

    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["origin_freq"] = df["Origin"].map(_freq_maps["origin_freq"]).fillna(0.0).to_numpy()
    X["dest_freq"] = df["Dest"].map(_freq_maps["dest_freq"]).fillna(0.0).to_numpy()
    X["route_freq"] = route.map(_freq_maps["route_freq"]).fillna(0.0).to_numpy()
    X["carrier_freq"] = df["UniqueCarrier"].map(_freq_maps["carrier_freq"]).fillna(0.0).to_numpy()

    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[NUM_COLS + TIME_COLS + cat_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.9,
    colsample_bytree=0.4,
    min_child_weight=40,
    reg_lambda=10.0,
    max_cat_threshold=64,
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
