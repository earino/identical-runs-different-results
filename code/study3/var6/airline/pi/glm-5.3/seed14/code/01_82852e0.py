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

# --- feature engineering ---------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
cat_levels = {
    c: pd.Index(sorted(train[c].dropna().astype(str).unique()))
    for c in ["UniqueCarrier", "Origin", "Dest"]
}
cat_levels["Route"] = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here; predict_proba() calls it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    # calendar: c-<n> strings -> ints
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = pd.to_numeric(df[c].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    # scheduled departure time (hhmm) -> hour / minute / minutes since midnight + cyclic
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24)
    minute = (dep % 100).clip(0, 59)
    mins = hour * 60 + minute
    X["DepHour"] = hour
    X["DepMinutes"] = mins
    frac = (mins % 1440) / 1440.0
    X["sin_time"] = np.sin(2 * np.pi * frac)
    X["cos_time"] = np.cos(2 * np.pi * frac)
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_distance"] = np.log1p(dist)
    # categoricals (route = origin_dest interaction)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in CAT_COLS:
        vals = df[c] if c != "Route" else route
        X[c] = pd.Categorical(vals, categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


BASE_PARAMS = dict(
    max_depth=8,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# --- internal early stopping on a 2005 holdout ----------------------------------
t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
rng = np.random.RandomState(SEED)
va_idx = rng.choice(len(X_all), size=10_000, replace=False)
mask = np.ones(len(X_all), dtype=bool)
mask[va_idx] = False
es = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=60, eval_metric="auc", **BASE_PARAMS)
es.fit(X_all[mask], y_all[mask], eval_set=[(X_all[~mask], y_all[~mask])], verbose=False)
best_iter = es.best_iteration + 1
print(f"ES rounds: {best_iter}, internal AUC: {es.best_score:.4f}")

# --- final model on the full training data --------------------------------------
model = xgb.XGBClassifier(n_estimators=best_iter, **BASE_PARAMS)
model.fit(X_all, y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
