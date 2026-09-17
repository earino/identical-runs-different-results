"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

_dep_hour = lambda s: (np.floor(s.astype(float) / 100).astype(int) % 24).astype(str)
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))
hour_dow_levels = pd.Index(sorted((_dep_hour(train["DepTime"]) + "_" + train["DayOfWeek"].astype(str)).unique()))
hour_carrier_levels = pd.Index(sorted((_dep_hour(train["DepTime"]) + "_" + train["UniqueCarrier"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # engineered: route pair as its own categorical
    X["Route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels)
    # cyclic month/day/dow
    m = df["Month"].str.slice(1).astype(int)
    d = df["DayofMonth"].str.slice(1).astype(int)
    w = df["DayOfWeek"].str.slice(1).astype(int)
    X["month_sin"] = np.sin(2 * np.pi * m / 12);  X["month_cos"] = np.cos(2 * np.pi * m / 12)
    X["day_sin"] = np.sin(2 * np.pi * d / 31);    X["day_cos"] = np.cos(2 * np.pi * d / 31)
    X["dow_sin"] = np.sin(2 * np.pi * w / 7);     X["dow_cos"] = np.cos(2 * np.pi * w / 7)
    # time of day: raw minutes + cyclic
    dep = df["DepTime"].astype(float)
    X["Dep_hour"] = np.floor(dep / 100).astype(int)
    X["Dep_min"] = dep - 100 * np.floor(dep / 100)
    X["dep_sin"] = np.sin(2 * np.pi * dep / 2400)
    X["dep_cos"] = np.cos(2 * np.pi * dep / 2400)
    # interactions as categoricals
    X["hour_x_dow"] = pd.Categorical(_dep_hour(df["DepTime"]) + "_" + df["DayOfWeek"].astype(str), categories=hour_dow_levels)
    X["hour_x_carrier"] = pd.Categorical(_dep_hour(df["DepTime"]) + "_" + df["UniqueCarrier"].astype(str), categories=hour_carrier_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=30,
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
