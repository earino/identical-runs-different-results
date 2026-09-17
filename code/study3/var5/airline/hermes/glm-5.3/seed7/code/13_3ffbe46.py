"""XGBoost airline-delay classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
hour_levels = [str(i) for i in range(25)]
# count encodings, fit on train only
route_counts = (train["Origin"].astype(str) + "-" + train["Dest"].astype(str)).value_counts().to_dict()
origin_counts = train["Origin"].value_counts().to_dict()
dest_counts = train["Dest"].value_counts().to_dict()
# schedule-pressure counts (target-agnostic, fit on train only): how many train flights
# share this origin & 15-min block / origin & hour / block
_mins_tr = (train["DepTime"].astype(int) // 100) * 60 + train["DepTime"].astype(int) % 100
bucket15_levels = pd.Index(sorted(((_mins_tr // 15) * 15).astype(str).unique()))
_ob = pd.DataFrame({"o": train["Origin"], "b": (_mins_tr // 15) * 15})
orig_block_counts = _ob.value_counts().to_dict()
_oh = pd.DataFrame({"o": train["Origin"], "h": _mins_tr // 60})
orig_hour_counts = _oh.value_counts().to_dict()
block_counts = ((_mins_tr // 15) * 15).value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # cyclic day-of-year from Month/DayofMonth (c-<n> strings)
    month = df["Month"].str.slice(2).astype(float)
    day = df["DayofMonth"].str.slice(2).astype(float)
    doy = (month - 1.0) * 30.4 + day
    X["SeasonSin"] = np.sin(2 * np.pi * doy / 365.0)
    X["SeasonCos"] = np.cos(2 * np.pi * doy / 365.0)
    # hour-of-day as categorical (delay risk is strongly time-of-day shaped)
    hour = np.floor(df["DepTime"].astype(float) / 100.0).clip(0, 24).astype(int).astype(str)
    X["DepHour"] = pd.Categorical(hour, categories=hour_levels)
    # count/frequency encodings (target-agnostic, fit on train only)
    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    X["RouteFreq"] = np.log1p(route.map(route_counts).fillna(0.0))
    X["OriginFreq"] = np.log1p(df["Origin"].map(origin_counts).fillna(0.0))
    X["DestFreq"] = np.log1p(df["Dest"].map(dest_counts).fillna(0.0))
    # true minutes since midnight (DepTime is hhmm: 1159 and 1200 are 1 min apart, not 41)
    hhmm = df["DepTime"].astype(int)
    mins = (hhmm // 100) * 60 + hhmm % 100
    X["DepMinutes"] = mins.astype(float)
    X["DepBucket15"] = pd.Categorical(((mins // 15) * 15).astype(str), categories=bucket15_levels)
    # schedule pressure at this departure time/place (train-fit counts)
    b15 = (mins // 15) * 15
    X["OrigBlockCount"] = [orig_block_counts.get((o, b), 0) for o, b in zip(df["Origin"], b15)]
    X["OrigHourCount"] = [orig_hour_counts.get((o, h), 0) for o, h in zip(df["Origin"], mins // 60)]
    X["BlockCount"] = [block_counts.get(b, 0) for b in b15]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# small heterogeneous ensemble of XGBoost models: different seeds and feature
# subsets (dropping one informative block per model) decorrelates errors.
ENSEMBLE = [
    (42, None),
    (7, ["Distance"]),
    (2024, ["SeasonSin", "SeasonCos"]),
    (13, ["Dest"]),
    (99, None),
]
_all_cols = prepare(train).columns.tolist()
models = []
for seed, drop in ENSEMBLE:
    feats = [c for c in _all_cols if c not in (drop or [])]
    models.append(
        (
            xgb.XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.025,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                enable_categorical=True,
                random_state=seed,
                n_jobs=N_JOBS,
            ),
            feats,
        )
    )

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
for m, feats in models:
    m.fit(X_train[feats], y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[feats])[:, 1] for m, feats in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
