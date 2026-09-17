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
_tr_hour = (pd.to_numeric(train["DepTime"], errors="coerce") // 100 % 24).astype(int)
carrier_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _tr_hour.astype(str)).unique()))
origin_hour_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + _tr_hour.astype(str)).unique()))
dest_hour_levels = pd.Index(sorted((train["Dest"].astype(str) + "_" + _tr_hour.astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dt = pd.to_numeric(df["DepTime"], errors="coerce").to_numpy()
    hour = ((dt // 100) % 24).astype(int)
    minute = (dt % 100).astype(int)
    tod = hour * 60 + np.where(minute >= 60, 59, minute)
    X["dep_hour"] = pd.Categorical(hour, categories=list(range(24)))
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    travel_rush = ((month == 12) & (dom >= 18)) | ((month == 1) & (dom <= 4)) | \
                  (((month == 6) & (dom >= 28)) | ((month == 7) & (dom <= 8))) | ((month == 11) & dom.between(20, 30))
    holiday_day = ((month == 11) & (dow == 4) & dom.between(22, 28)) | ((month == 5) & (dow == 1) & (dom >= 25)) | \
                  ((month == 9) & (dow == 1) & (dom <= 7)) | ((month == 7) & (dom == 4)) | \
                  ((month == 12) & (dom == 25)) | ((month == 1) & (dom == 1))
    X["travel_rush"] = travel_rush.astype(np.int8).to_numpy()
    X["holiday_day"] = holiday_day.astype(np.int8).to_numpy()
    X["is_weekend"] = (dow >= 6).astype(np.int8).to_numpy()
    ch = df["UniqueCarrier"].astype(str) + "_" + pd.Series(hour, index=df.index).astype(str)
    X["carrier_hour"] = pd.Categorical(ch, categories=carrier_hour_levels)
    oh = df["Origin"].astype(str) + "_" + pd.Series(hour, index=df.index).astype(str)
    X["origin_hour"] = pd.Categorical(oh, categories=origin_hour_levels)
    dh = df["Dest"].astype(str) + "_" + pd.Series(hour, index=df.index).astype(str)
    X["dest_hour"] = pd.Categorical(dh, categories=dest_hour_levels)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    (200, 4, 0.05, 0.9, 0.9, 42),
    (300, 5, 0.04, 0.9, 0.9, 202),
    (150, 3, 0.07, 0.9, 0.9, 777),
    (400, 4, 0.03, 0.8, 0.8, 1234),
    (200, 6, 0.05, 0.9, 0.9, 55),
]
models = []

t0 = time.time()
X_train = prepare(train)
y_ = to_y(train)
for n, d, lr, ss, cs, s in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=n,
        max_depth=d,
        learning_rate=lr,
        subsample=ss,
        colsample_bytree=cs,
        tree_method="hist",
        enable_categorical=True,
        reg_lambda=5.0,
        reg_alpha=5.0,
        sampling_method="gradient_based",
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
