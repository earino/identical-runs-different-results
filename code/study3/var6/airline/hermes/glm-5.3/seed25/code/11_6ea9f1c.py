"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# raw string columns are re-encoded numerically in prepare(); only these stay categorical
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
feature_cols = ["DepTime", "Distance"] + cat_cols + ["Month", "DayofMonth", "DayOfWeek"]


# --- train-only count statistics (unsupervised, no target leakage) -------------
hour_tr = (train["DepTime"].astype(int) // 100).astype(str)
blk_tr = (((train["DepTime"].astype(int) // 100) * 60 + train["DepTime"].astype(int) % 100) // 30).astype(str)
route_tr = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
cnt_origin_hour = train.groupby(train["Origin"].astype(str) + "-" + hour_tr).size()
cnt_dest_hour = train.groupby(train["Dest"].astype(str) + "-" + hour_tr).size()
cnt_route = train.groupby(train["Origin"].astype(str) + "-" + train["Dest"].astype(str)).size()
cnt_origin = train.groupby(train["Origin"].astype(str)).size()
cnt_dest = train.groupby(train["Dest"].astype(str)).size()
cnt_route_hour = train.groupby(route_tr + "-" + hour_tr).size()
cnt_route_blk = train.groupby(route_tr + "-" + blk_tr).size()
cnt_hour = train.groupby(hour_tr).size()
cnt_carrier_hour = train.groupby(train["UniqueCarrier"].astype(str) + "-" + hour_tr).size()
cnt_carrier_route = train.groupby(train["UniqueCarrier"].astype(str) + "-" + route_tr).size()
med_route_dist = train.groupby(route_tr)["Distance"].median()
GLOBAL_DIST_MED = float(train["Distance"].median())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    # numeric time features from DepTime (hhmm)
    dt = X["DepTime"].astype(int)
    hour = dt // 100
    minute = dt % 100
    dep_min = hour * 60 + minute                      # minutes since midnight
    X["DepTime_min"] = dep_min
    X["DepTime_h"] = hour
    X["DepTime_blk30"] = dep_min // 30                # 48 blocks of 30 minutes
    X["sin_t"] = np.sin(2 * np.pi * dep_min / 1440.0)
    X["cos_t"] = np.cos(2 * np.pi * dep_min / 1440.0)
    X["night"] = (dep_min < 6 * 60).astype(int)       # red-eye / very early flights
    # numeric calendar features from c-<n> strings (raw c-strings dropped as categoricals)
    for col, name in (("Month", "mon"), ("DayofMonth", "dom"), ("DayOfWeek", "dow")):
        X[name] = X[col].str.slice(2).astype(int)
    X["sin_mon"] = np.sin(2 * np.pi * X["mon"] / 12.0)
    X["cos_mon"] = np.cos(2 * np.pi * X["mon"] / 12.0)
    X["sin_dow"] = np.sin(2 * np.pi * X["dow"] / 7.0)
    X["cos_dow"] = np.cos(2 * np.pi * X["dow"] / 7.0)
    X = X.drop(columns=["Month", "DayofMonth", "DayOfWeek"])  # raw c-strings: numeric versions kept
    # traffic counts (fit on training data only; unseen combos -> 0)
    h = hour.astype(str)
    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    X["n_oh"] = (df["Origin"].astype(str) + "-" + h).map(cnt_origin_hour).fillna(0).to_numpy()
    X["n_dh"] = (df["Dest"].astype(str) + "-" + h).map(cnt_dest_hour).fillna(0).to_numpy()
    X["n_route"] = route.map(cnt_route).fillna(0).to_numpy()
    X["n_origin"] = df["Origin"].astype(str).map(cnt_origin).fillna(0).to_numpy()
    X["n_dest"] = df["Dest"].astype(str).map(cnt_dest).fillna(0).to_numpy()
    X["dist_vs_route_med"] = X["Distance"] - route.map(med_route_dist).fillna(GLOBAL_DIST_MED).to_numpy()
    X["n_route_h"] = (route + "-" + h).map(cnt_route_hour).fillna(0).to_numpy()
    X["n_route_blk"] = (route + "-" + (dep_min // 30).astype(str)).map(cnt_route_blk).fillna(0).to_numpy()
    X["n_hour"] = h.map(cnt_hour).fillna(0).to_numpy()
    X["n_carrier_h"] = (df["UniqueCarrier"].astype(str) + "-" + h).map(cnt_carrier_hour).fillna(0).to_numpy()
    X["n_carrier_route"] = (df["UniqueCarrier"].astype(str) + "-" + route).map(cnt_carrier_route).fillna(0).to_numpy()
    X["log_n_oh"] = np.log1p(X["n_oh"])
    X["log_n_dh"] = np.log1p(X["n_dh"])
    X["log_n_route_h"] = np.log1p(X["n_route_h"])
    X["oh_over_o"] = X["n_oh"] / (X["n_origin"] + 1.0)
    X["dh_over_d"] = X["n_dh"] / (X["n_dest"] + 1.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.04,
    subsample=0.7,
    colsample_bytree=0.3,
    min_child_weight=20,
    reg_lambda=10.0,
    max_bin=1024,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = (42, 7, 123)

t0 = time.time()
models = []
for sd in SEEDS:
    m = xgb.XGBClassifier(random_state=sd, **PARAMS)
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
