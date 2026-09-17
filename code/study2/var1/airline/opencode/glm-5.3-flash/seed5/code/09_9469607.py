"""XGBoost binary classifier on the airline dataset. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
freq_maps = {c: train[c].value_counts(normalize=True) for c in CAT_COLS}
_tr_route = train["Origin"].astype(str) + ">" + train["Dest"].astype(str)
route_freq = _tr_route.value_counts(normalize=True)
hub_dest = train.groupby("Origin")["Dest"].nunique()
hub_orig = train.groupby("Dest")["Origin"].nunique()
hub_carrier = train.groupby("UniqueCarrier")["Origin"].nunique()
route_dist = train.assign(_r=_tr_route).groupby("_r")["Distance"].agg(["mean", "std"])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayOfMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    dt = df["DepTime"].astype(int)
    hour, minute = dt // 100, dt % 100
    X["DepHour"] = hour
    X["DepMinute"] = minute
    ang = 2 * np.pi * (hour * 60 + minute) / 1440.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    X["DepTime"] = dt
    X["Distance"] = df["Distance"].astype(float)
    X["DistanceLog"] = np.log1p(df["Distance"].astype(float))
    route = df["Origin"].astype(str) + ">" + df["Dest"].astype(str)
    rd_mean = route.map(route_dist["mean"]).astype(float)
    rd_std = route.map(route_dist["std"]).astype(float)
    X["DistVsRoute"] = (X["Distance"] - rd_mean) / rd_std.clip(lower=1.0)
    X["HubOrigin"] = df["Origin"].map(hub_dest).astype(float).fillna(1.0)
    X["HubDest"] = df["Dest"].map(hub_orig).astype(float).fillna(1.0)
    X["HubCarrier"] = df["UniqueCarrier"].map(hub_carrier).astype(float).fillna(1.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
        X[c + "Freq"] = df[c].map(freq_maps[c]).astype(float).fillna(0.0)
    X["RouteFreq"] = route.map(route_freq).astype(float).fillna(0.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
X_full, y_full = prepare(train), to_y(train)

MEMBERS = [
    dict(seed=42, subsample=0.8, colsample_bytree=0.8, max_depth=8),
    dict(seed=7, subsample=0.7, colsample_bytree=0.9, max_depth=8),
    dict(seed=123, subsample=0.9, colsample_bytree=0.7, max_depth=7),
    dict(seed=2026, subsample=0.8, colsample_bytree=0.8, max_depth=12),
]
BASE = dict(learning_rate=0.05, tree_method="hist", enable_categorical=True)


def make_params(n_estimators: int, m: dict) -> dict:
    return dict(n_estimators=n_estimators, **{**BASE, **{k: v for k, v in m.items() if k != "seed"},
                                             "random_state": m["seed"], "n_jobs": N_JOBS})


# NUM_TREES was selected by early stopping (patience 50, 80/20 stratified split, seed 42); the selection is
# deterministic for this data/feature set, so hardcoding it avoids re-running the ES sweep on every run.
NUM_TREES = 1413

models = []
for m in MEMBERS:
    mdl = xgb.XGBClassifier(**make_params(NUM_TREES, m))
    mdl.fit(X_full, y_full)
    print(f"member seed={m['seed']} d={m['max_depth']} sub={m['subsample']} col={m['colsample_bytree']} done")
    models.append(mdl)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([mdl.predict_proba(prepare(df))[:, 1] for mdl in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
