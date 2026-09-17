"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Notes:
  - Plain categoricals (carrier/origin/dest) + numeric time features beat target encodings here.
  - All feature engineering lives inside fe(); encoders/stats are fit on train only.
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
HOUR_LEVELS = pd.Index(range(25))
B15_LEVELS = pd.Index(range(97))  # 15-minute departure-time buckets (0..96)


def fe(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls fe() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.slice(2).astype(int)
    X["day"] = df["DayofMonth"].str.slice(2).astype(int)
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    h = (df["DepTime"] // 100).clip(0, 24)
    mins = h * 60 + df["DepTime"] % 100
    X["dep_raw"] = df["DepTime"].astype(int)
    X["hour"] = h
    X["hour_c"] = h  # converted to categorical in prepare()
    X["b15"] = (mins // 15).clip(0, 96)  # 15-minute buckets, categorical in prepare()
    X["mins"] = mins
    X["sin"] = np.sin(2 * np.pi * mins / 1440)
    X["cos"] = np.cos(2 * np.pi * mins / 1440)
    X["minofhr"] = df["DepTime"] % 100
    X["dist"] = df["Distance"].astype(float)
    X["logdist"] = np.log1p(df["Distance"].astype(float))
    for c in CAT_COLS:
        X[c] = df[c].to_numpy()
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = fe(df)
    for c in CAT_COLS + ["hour_c", "b15"]:
        if c == "hour_c":
            X[c] = pd.Categorical(X[c], categories=HOUR_LEVELS)
        elif c == "b15":
            X[c] = pd.Categorical(X[c], categories=B15_LEVELS)
        else:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Two XGBoost models (different seeds/colsample) averaged: a small ensemble reduces
# variance vs a single refit at the same wall-clock cost as the old ES+refit combo.
ENSEMBLE = [
    dict(n_estimators=400, colsample_bytree=0.6, seed=0),
    dict(n_estimators=400, colsample_bytree=0.5, seed=1),
    dict(n_estimators=320, colsample_bytree=0.6, seed=2, subsample=0.7),
]


def _make_model(n_estimators, colsample_bytree, seed, subsample=0.8):
    return xgb.XGBClassifier(
        max_depth=20,
        learning_rate=0.02,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=1,
        reg_lambda=1.0,
        reg_alpha=0.5,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + seed,
        n_jobs=N_JOBS,
    ).set_params(n_estimators=n_estimators)


t0 = time.time()
Xp = prepare(train)
y = to_y(train)
models = []
for spec in ENSEMBLE:
    m = _make_model(spec["n_estimators"], spec["colsample_bytree"], spec["seed"], spec.get("subsample", 0.8))
    m.fit(Xp, y)
    models.append(m)
    print(f"member trained, time {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xq = prepare(df)
    return np.mean([m.predict_proba(Xq)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
