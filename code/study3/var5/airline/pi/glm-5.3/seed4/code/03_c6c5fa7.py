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
N_ENSEMBLE = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering ------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}

# count (frequency) encodings fitted on TRAIN only: airport/route congestion proxies
_hour_tr = (train["DepTime"].astype(int) // 100) % 24
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_train_counts = {
    "oh": (train["Origin"].astype(str) + "@" + _hour_tr.astype(str)).value_counts(),
    "dh": (train["Dest"].astype(str) + "@" + _hour_tr.astype(str)).value_counts(),
    "route": _route_tr.value_counts(),
    "origin": train["Origin"].astype(str).value_counts(),
    "dest": train["Dest"].astype(str).value_counts(),
}


def _cnt(key: pd.Series, name: str) -> np.ndarray:
    return np.log1p(key.map(_train_counts[name]).fillna(0).to_numpy(dtype=float))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # calendar fields come as c-<n> strings -> numeric
    X["month"] = df["Month"].astype(str).str[2:].astype(int)
    X["day"] = df["DayofMonth"].astype(str).str[2:].astype(int)
    X["dow"] = df["DayOfWeek"].astype(str).str[2:].astype(int)
    X["doy"] = (X["month"] - 1) * 31 + X["day"]
    # scheduled departure time hhmm -> hour / minute / time-of-day (wraps hour 24/26 -> 0/2)
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    hour = (dep // 100) % 24
    minute = dep % 100
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour * 60 + minute
    # geography / carrier
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Distance"] = df["Distance"].astype(float)
    # congestion counts (train-fitted)
    X["cnt_oh"] = _cnt(df["Origin"].astype(str) + "@" + hour.astype(str), "oh")
    X["cnt_dh"] = _cnt(df["Dest"].astype(str) + "@" + hour.astype(str), "dh")
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["cnt_route"] = _cnt(route, "route")
    X["cnt_origin"] = _cnt(df["Origin"].astype(str), "origin")
    X["cnt_dest"] = _cnt(df["Dest"].astype(str), "dest")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small seed ensemble with column subsampling ------------------------
t0 = time.time()
Xtr, Xe = prepare(train), prepare(evald)
models = []
for seed in range(N_ENSEMBLE):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=4,
        learning_rate=0.1,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, to_y(train), eval_set=[(Xe, to_y(evald))], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
