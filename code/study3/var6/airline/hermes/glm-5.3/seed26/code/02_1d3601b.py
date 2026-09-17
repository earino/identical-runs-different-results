"""Airline dep-delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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


def _num(s: pd.Series) -> pd.Series:
    """'c-7' -> 7  (Month, DayofMonth, DayOfWeek arrive as c-<n> strings)."""
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


# raw numeric columns that stay numeric
raw_num_cols = ["DepTime", "Distance"]
# Month/DayofMonth/DayOfWeek: convert to true ints (numeric ordering helps splits)
cal_num_cols = ["Month", "DayofMonth", "DayOfWeek"]
cat_cols = ["UniqueCarrier", "Origin", "Dest"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# route distance statistics (label-free, from train only)
_train_route = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
_dist_stats = train.assign(Route=_train_route).groupby("Route")["Distance"].agg(["mean", "count"])
route_mean_dist = _dist_stats["mean"]
route_count = _dist_stats["count"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in cal_num_cols:
        X[c] = _num(df[c]).astype("float32")
    for c in raw_num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    # DepTime hhmm -> hour/minute parts
    dep = X["DepTime"]
    X["DepHour"] = np.floor(dep / 100).astype("float32")
    X["DepMinute"] = (dep % 100).astype("float32")
    # cyclic time-of-day position: minutes since midnight
    mins = (X["DepHour"] * 60 + X["DepMinute"]).astype("float32")
    X["DepSin"] = np.sin(2 * np.pi * mins / 1440).astype("float32")
    X["DepCos"] = np.cos(2 * np.pi * mins / 1440).astype("float32")
    # cyclic calendar features
    X["MonthSin"] = np.sin(2 * np.pi * X["Month"] / 12).astype("float32")
    X["MonthCos"] = np.cos(2 * np.pi * X["Month"] / 12).astype("float32")
    X["DowSin"] = np.sin(2 * np.pi * X["DayOfWeek"] / 7).astype("float32")
    X["DowCos"] = np.cos(2 * np.pi * X["DayOfWeek"] / 7).astype("float32")
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    # route distance stats (label-free)
    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    X["route_mean_dist"] = route.map(route_mean_dist).astype("float32")
    X["route_count"] = route.map(route_count).astype("float32")
    X["dist_vs_route"] = (X["Distance"] - X["route_mean_dist"]).astype("float32")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=30,
    eval_metric="auc",
)

t0 = time.time()
Xtr = prepare(train)
model.fit(Xtr, to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
