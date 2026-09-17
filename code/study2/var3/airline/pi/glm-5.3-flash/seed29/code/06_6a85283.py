"""XGBoost binary classifier for airline delays. Contract: see program.md.

Exp 5: baseline model config + engineered time features (hour, minute, cyclic
time-of-day, month/dow as ints, log distance, night flag).
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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# smoothed target encodings, fit on TRAIN ONLY, applied inside prepare()
y_bin = (train[TARGET] == POSITIVE).astype(float)
GLOBAL_MEAN = float(y_bin.mean())
TE_K = {"UniqueCarrier": 20, "Origin": 100, "Dest": 100}
te_maps = {}
for c, k in TE_K.items():
    stats = y_bin.groupby(train[c]).agg(["mean", "count"])
    te_maps[c] = ((stats["count"] * stats["mean"] + k * GLOBAL_MEAN) / (stats["count"] + k))

# interaction keys are built the same way in prepare(); encodings fit on train only
IX_K = {"route": 200, "carrier_hour": 50, "origin_hour": 50}


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    hr = ((pd.to_numeric(df["DepTime"]) // 100).clip(0, 47)).astype(int)
    return pd.DataFrame({
        "route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "carrier_hour": df["UniqueCarrier"].astype(str) + "_" + hr.astype(str),
        "origin_hour": df["Origin"].astype(str) + "_" + hr.astype(str),
    }, index=df.index)


_ktr = _keys(train)
ix_maps = {}
for c, k in IX_K.items():
    stats = y_bin.groupby(_ktr[c]).agg(["mean", "count"])
    ix_maps[c] = ((stats["count"] * stats["mean"] + k * GLOBAL_MEAN) / (stats["count"] + k))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c, m in te_maps.items():
        X[c + "_te"] = df[c].map(m).astype(float).fillna(GLOBAL_MEAN)
    kd = _keys(df)
    for c, m in ix_maps.items():
        X[c + "_te"] = kd[c].map(m).astype(float).fillna(GLOBAL_MEAN)
    # engineered numeric features (all derived from the row itself, nothing fit on data)
    dt = pd.to_numeric(df["DepTime"])
    hour = (dt // 100).astype(float)
    minute = (dt % 100).astype(float)
    tmin = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tmin / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tmin / 1440)
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"]))
    X["is_night"] = ((hour >= 21) | (hour <= 5)).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
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
