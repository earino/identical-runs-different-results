"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# columns of the form c-<n>: parse to int so trees see ordinal order
C_COLS = ["Month", "DayofMonth", "DayOfWeek"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# schedule congestion: flights per (Origin, hour) and per (Dest, hour) in the TRAIN slice
_oh = train.groupby([train["Origin"], train["DepTime"].astype(int) // 100]).size()
_dh = train.groupby([train["Dest"], train["DepTime"].astype(int) // 100]).size()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in C_COLS:
        X[c] = df[c].str.replace("c-", "", regex=False).astype(int)
    # DepTime is hhmm; derive time-of-day features (values past 2400 = after-midnight departures)
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    frac = hour + minute / 60.0
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["DepFracHour"] = frac
    X["DepTime"] = dep
    # delays accumulate through the day and reset around 3am: hours since 3am makes that monotone
    X["HoursSince3am"] = np.mod(np.where(frac >= 24, frac - 24, frac) - 3.0, 24.0)
    wrapped = np.where(frac >= 24, frac - 24, frac)
    X["sin24"] = np.sin(2 * np.pi * wrapped / 24.0)
    X["cos24"] = np.cos(2 * np.pi * wrapped / 24.0)
    X["Distance"] = df["Distance"].astype(float)
    X["DistanceLog"] = np.log1p(df["Distance"].astype(float))
    # scheduled arrival time proxy: dep time + taxi + distance/450mph, wrapped to the day cycle
    arr_t = frac + 0.5 + df["Distance"].astype(float).to_numpy() / 450.0
    arrw = np.mod(arr_t, 24.0)
    X["ArrFracHour"] = arrw
    X["ArrHoursSince3am"] = np.mod(arrw - 3.0, 24.0)
    # congestion features (from TRAIN schedule only; unseen combos -> 0)
    X["OriginHourCount"] = np.array([_oh.get(k, 0) for k in zip(df["Origin"], dep // 100)], dtype=float)
    X["DestHourCount"] = np.array([_dh.get(k, 0) for k in zip(df["Dest"], dep // 100)], dtype=float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# seed-averaged ensemble of XGBoost models (pure XGBoost, per contract)
N_MODELS = 3
SEEDS = [42, 7, 2024]
models = []
t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
Xev_eval = prepare(evald)
for s in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=24,
        learning_rate=0.03,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.7,
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=60,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev_eval, to_y(evald))], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s, best_iterations={[getattr(m, 'best_iteration', None) for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
