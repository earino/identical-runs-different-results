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
# frequency statistics fitted on training data only
_train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
freq_maps = {
    "Origin": train["Origin"].value_counts(normalize=True),
    "Dest": train["Dest"].value_counts(normalize=True),
    "UniqueCarrier": train["UniqueCarrier"].value_counts(normalize=True),
    "Route": _train_route.value_counts(normalize=True),
}


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

    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["origin_freq"] = df["Origin"].map(freq_maps["Origin"]).astype(float).to_numpy()
    X["dest_freq"] = df["Dest"].map(freq_maps["Dest"]).astype(float).to_numpy()
    X["carrier_freq"] = df["UniqueCarrier"].map(freq_maps["UniqueCarrier"]).astype(float).to_numpy()
    X["route_freq"] = route.map(freq_maps["Route"]).astype(float).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=4000,
    max_depth=16,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
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
