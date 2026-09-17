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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- route/traffic features, fit on TRAIN ONLY ---------------------------------
org_stats = train.groupby("Origin").agg(
    org_n=("Distance", "size"), org_med_dist=("Distance", "median")
)
dest_stats = train.groupby("Dest").agg(
    dest_n=("Distance", "size"), dest_med_dist=("Distance", "median")
)
route_dist = train.groupby(["Origin", "Dest"])["Distance"].median().rename("route_med_dist")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # c-<n> strings -> ordered integers
    X["DayofMonth"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    X["Hour"] = dep // 100
    X["Minute"] = dep % 100
    X["Distance"] = df["Distance"].astype(float)
    X["log_Distance"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    # route / airport traffic features (fit on train only)
    X["same_airport"] = (df["Origin"] == df["Dest"]).astype(int)
    X["org_n"] = df["Origin"].map(org_stats["org_n"]).fillna(0)
    X["dest_n"] = df["Dest"].map(dest_stats["dest_n"]).fillna(0)
    key = pd.MultiIndex.from_arrays([df["Origin"], df["Dest"]])
    X["route_med_dist"] = key.map(route_dist).astype(float)
    X["dist_vs_route"] = X["Distance"] - X["route_med_dist"]
    X["dist_vs_route_rel"] = X["dist_vs_route"] / (X["route_med_dist"] + 1.0)
    X["org_med_dist"] = df["Origin"].map(org_stats["org_med_dist"]).astype(float)
    X["dest_med_dist"] = df["Dest"].map(dest_stats["dest_med_dist"]).astype(float)
    X["dist_vs_org_med"] = X["Distance"] - X["org_med_dist"]
    X["dist_vs_dest_med"] = X["Distance"] - X["dest_med_dist"]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=6000,
    max_depth=24,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=200,
    n_jobs=N_JOBS,
)

t0 = time.time()
rng = np.random.RandomState(SEED)
tr_idx = rng.rand(len(train)) < 0.9
Xva, yva = prepare(train[~tr_idx]), to_y(train[~tr_idx])
Xall, yall = prepare(train), to_y(train)

# stage 1: one ES run to find the iteration count
m0 = xgb.XGBClassifier(random_state=SEED, **PARAMS)
m0.fit(Xall[tr_idx], yall[tr_idx], eval_set=[(Xva, yva)], verbose=False)
n_best = int(m0.best_iteration + 1)
print(f"Stage 1: {time.time() - t0:.1f}s, best iter: {n_best - 1}")

# stage 2: refit on ALL training rows with the tuned iteration count, across seeds
models = []
N_SEEDS = 2
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(random_state=SEED + 1000 * s, **PARAMS)
    m.set_params(n_estimators=n_best, early_stopping_rounds=None)
    m.fit(Xall, yall, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s, {len(models)} models @ {n_best} trees")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
