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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X = _add_freq(X)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- frequency encoding (label-free, fit on TRAINING data only) ---------------
def _freq_map(series: pd.Series) -> dict:
    return series.astype(str).value_counts().to_dict()


_train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
FR_ORIGIN = _freq_map(train["Origin"])
FR_DEST = _freq_map(train["Dest"])
FR_CARRIER = _freq_map(train["UniqueCarrier"])
FR_ROUTE = _freq_map(_train_route)

_th = (train["DepTime"].astype("int64") // 100).astype(str)
FR_ORIGIN_HOUR = (train["Origin"].astype(str) + "|" + _th).value_counts().to_dict()
FR_DEST_HOUR = (train["Dest"].astype(str) + "|" + _th).value_counts().to_dict()
FR_CARRIER_ORIGIN = (train["UniqueCarrier"].astype(str) + "|" + train["Origin"].astype(str)).value_counts().to_dict()
FR_CARRIER_HOUR = (train["UniqueCarrier"].astype(str) + "|" + _th).value_counts().to_dict()


def _add_freq(X: pd.DataFrame) -> pd.DataFrame:
    X["fr_origin"] = X["Origin"].astype(str).map(FR_ORIGIN).fillna(0)
    X["fr_dest"] = X["Dest"].astype(str).map(FR_DEST).fillna(0)
    X["fr_carrier"] = X["UniqueCarrier"].astype(str).map(FR_CARRIER).fillna(0)
    route = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    X["fr_route"] = route.map(FR_ROUTE).fillna(0)
    th = (X["DepTime"].astype("int64") // 100).astype(str)
    X["fr_origin_hour"] = (X["Origin"].astype(str) + "|" + th).map(FR_ORIGIN_HOUR).fillna(0)
    X["fr_dest_hour"] = (X["Dest"].astype(str) + "|" + th).map(FR_DEST_HOUR).fillna(0)
    X["fr_carrier_origin"] = (X["UniqueCarrier"].astype(str) + "|" + X["Origin"].astype(str)).map(FR_CARRIER_ORIGIN).fillna(0)
    X["ratio_origin_hour"] = X["fr_origin_hour"] / X["fr_origin"].clip(lower=1)
    X["ratio_dest_hour"] = X["fr_dest_hour"] / X["fr_dest"].clip(lower=1)
    X["ratio_carrier_origin"] = X["fr_carrier_origin"] / X["fr_carrier"].clip(lower=1)
    car = X["UniqueCarrier"].astype(str)
    X["fr_carrier_hour"] = (car + "|" + th).map(FR_CARRIER_HOUR).fillna(0)
    X["ratio_carrier_hour"] = X["fr_carrier_hour"] / X["fr_carrier"].clip(lower=1)
    return X


# --- model --------------------------------------------------------------------
N_MODELS = 5
models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=10,
        reg_lambda=5.0,
        reg_alpha=0.5,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 100 * i,
        n_jobs=N_JOBS,
    )
    models.append(m)

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
