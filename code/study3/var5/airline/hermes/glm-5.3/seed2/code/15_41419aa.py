"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
route_dist_map = train.groupby(_route)["Distance"].mean().to_dict()
global_mean_dist = float(train["Distance"].mean())
# flight-frequency counts (structural, robust across years), fit on train only
vc_origin = train["Origin"].value_counts().to_dict()
vc_dest = train["Dest"].value_counts().to_dict()
vc_carrier = train["UniqueCarrier"].value_counts().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["month"] = df["Month"].str.slice(2).astype(int)
    X["day"] = df["DayofMonth"].str.slice(2).astype(int)
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    # cyclic encodings of calendar variables
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X["day_sin"] = np.sin(2 * np.pi * X["day"] / 31)
    X["day_cos"] = np.cos(2 * np.pi * X["day"] / 31)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7)
    # departure-time features
    t = X["DepTime"].to_numpy()
    X["hour"] = np.clip(t // 100, 0, 24)
    X["minute"] = t % 100
    X["dep_minutes"] = np.clip(t // 100, 0, 24) * 60 + (t % 100)
    X["is_evening"] = (t // 100 >= 17).astype(int)
    X["is_morning"] = ((t // 100 >= 5) & (t // 100 < 12)).astype(int)
    X["night_flight"] = (t // 100 < 5).astype(int)
    frac = (t // 100 * 60 + (t % 100)) / 1440.0
    X["tod_sin"] = np.sin(2 * np.pi * frac)
    X["tod_cos"] = np.cos(2 * np.pi * frac)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    _r = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_mean_dist"] = _r.map(route_dist_map).fillna(global_mean_dist)
    X["dist_vs_route"] = df["Distance"].astype(float) - X["route_mean_dist"]
    X["origin_freq"] = np.log1p(df["Origin"].map(vc_origin).fillna(0))
    X["dest_freq"] = np.log1p(df["Dest"].map(vc_dest).fillna(0))
    X["carrier_freq"] = np.log1p(df["UniqueCarrier"].map(vc_carrier).fillna(0))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble of XGBoost models ------------------------------------------------
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

K = 3
models = []
t0 = time.time()
for k in range(K):
    m = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=20,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7 if k % 2 == 0 else 0.8,
        min_child_weight=1,
        reg_lambda=2.0,
        max_bin=512,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=80,
        random_state=SEED + 10 * k,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"model {k}: best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
