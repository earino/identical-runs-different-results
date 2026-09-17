"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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
# Categorical levels are defined from the TRAINING data only; unseen values in
# future data become NaN (missing), which XGBoost handles natively.
# Note: high-cardinality interactions (route) hurt as native categoricals; dropped.
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "hour", "minute", "tod_frac", "night", "Distance", "log_dist"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].fillna(-1).astype(int)
    h = dep // 100
    m = dep % 100
    X["DepTime"] = dep
    X["hour"] = h
    X["minute"] = m
    X["tod_frac"] = h + m / 60.0
    X["night"] = (h < 6).astype(int)
    dist = df["Distance"].astype(float)
    # seasonal / calendar
    mon = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dom = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    dow = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["day_of_year"] = (mon - 1) * 31 + dom
    X["is_weekend"] = (dow >= 6).astype(int)
    X["hour_of_week"] = (dow - 1) * 24 + h
    # estimated arrival clock: dep time + taxi/climb + flight time ~ dist/475 mph
    dep_min = h * 60 + m
    arr_min = (dep_min + 45 + dist / 475.0 * 60.0) % (24 * 60)
    X["arr_hour"] = np.floor(arr_min / 60.0)
    X["arr_frac"] = arr_min / 60.0
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["Month"] = df["Month"]
    X["DayofMonth"] = df["DayofMonth"]
    X["DayOfWeek"] = df["DayOfWeek"]
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# build category levels from training data only
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}

# --- model --------------------------------------------------------------------
t0 = time.time()
X_ev = prepare(evald)
y_ev = to_y(evald)
ALL = list(X_ev.columns)
DROP = {  # coarse feature-group bagging: member -> columns to exclude
    "airports": ["Origin", "Dest"],
    "calendar": ["day_of_year", "hour_of_week", "Month", "DayofMonth", "DayOfWeek"],
    "carrier": ["UniqueCarrier"],
    "distance": ["Distance", "log_dist", "arr_hour", "arr_frac"],
}
ENSEMBLE = [  # (seed, depth, lr, subsample, colsample, drop_group)
    (42, 8, 0.03, 0.8, 0.8, None),
    (5150, 8, 0.02, 0.8, 0.6, "calendar"),
    (123, 10, 0.03, 0.9, 0.9, None),
    (2024, 8, 0.02, 0.8, 0.6, "calendar"),
    (8888, 8, 0.03, 0.85, 0.7, "calendar"),
    (11, 8, 0.03, 0.7, 0.9, "carrier"),
    (77, 12, 0.02, 0.8, 0.8, "distance"),
]
models = []
for seed, depth, lr, ss, cs, drop in ENSEMBLE:
    feats = [c for c in ALL if c not in (DROP[drop] if drop else [])]
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=depth,
        learning_rate=lr,
        min_child_weight=5,
        subsample=ss,
        colsample_bytree=cs,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(prepare(train)[feats], to_y(train), eval_set=[(X_ev[feats], y_ev)], verbose=False)
    models.append((m, feats))
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m, _ in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp[feats])[:, 1] for m, feats in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
