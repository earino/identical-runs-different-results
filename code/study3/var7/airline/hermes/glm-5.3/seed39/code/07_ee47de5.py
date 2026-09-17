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
cat_cols = obj_cols  # keep all string columns as categoricals
num_cols = [c for c in feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# flight-count (traffic volume) statistics from TRAIN only
vol_origin = train["Origin"].value_counts()
vol_dest = train["Dest"].value_counts()
vol_route = (train["Origin"] + "_" + train["Dest"]).value_counts()
vol_carrier = train["UniqueCarrier"].value_counts()
MED_VOL = float(vol_route.median())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    # numeric time features
    dep = df["DepTime"]
    hour = dep // 100
    minute = dep % 100
    frac_day = (hour * 60 + minute) / 1440.0
    X["hour"] = hour
    X["minute"] = minute
    X["frac_day"] = frac_day
    X["sin_h"] = np.sin(2 * np.pi * frac_day)
    X["cos_h"] = np.cos(2 * np.pi * frac_day)
    X["sin_month"] = np.sin(2 * np.pi * df["Month"].str[2:].astype(int) / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * df["Month"].str[2:].astype(int) / 12.0)
    X["sin_w"] = np.sin(2 * np.pi * df["DayOfWeek"].str[2:].astype(int) / 7.0)
    X["cos_w"] = np.cos(2 * np.pi * df["DayOfWeek"].str[2:].astype(int) / 7.0)
    # traffic volume features (train-side counts)
    X["vol_origin"] = np.log1p(df["Origin"].map(vol_origin).fillna(MED_VOL))
    X["vol_dest"] = np.log1p(df["Dest"].map(vol_dest).fillna(MED_VOL))
    X["vol_route"] = np.log1p((df["Origin"] + "_" + df["Dest"]).map(vol_route).fillna(0))
    X["vol_carrier"] = np.log1p(df["UniqueCarrier"].map(vol_carrier).fillna(MED_VOL))
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
)
N_MODELS = 8
DEPTHS = [6, 8, 10, 8, 6, 7, 9, 8]
SUBSAMPLES = [0.8, 0.8, 0.8, 0.7, 0.9, 0.75, 0.85, 0.8]
COLSAMPLES = [0.8, 0.8, 0.8, 0.9, 0.7, 0.85, 0.75, 0.6]
models = []
t0 = time.time()
Xev = prepare(evald)
for k in range(N_MODELS):
    m = xgb.XGBClassifier(
        **{**PARAMS,
           "random_state": SEED + 10 * k,
           "max_depth": DEPTHS[k],
           "subsample": SUBSAMPLES[k],
           "colsample_bytree": COLSAMPLES[k]},
    )
    m.fit(prepare(train), to_y(train), eval_set=[(Xev, to_y(evald))], verbose=False)
    models.append(m)
    print(f"model {k}: best_round={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
