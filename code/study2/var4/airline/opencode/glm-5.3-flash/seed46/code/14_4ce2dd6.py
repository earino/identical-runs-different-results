"""Airline delay XGBoost. v6: counts-only encoding (TE dropped after ablation).

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _key(df: pd.DataFrame, k: str) -> pd.Series:
    return _route(df) if k == "route" else df[k].astype(str)


# --- encoding statistics, fit on TRAIN only -------------------------------------


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # scheduled departure time (hhmm) -> time of day
    dt = _num(df["DepTime"])
    hour = (dt // 100).clip(0, 23)
    minute = dt - hour * 100
    tod = (hour * 60 + minute) / 1440.0
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    X["hour"] = hour
    X["red_eye"] = ((hour >= 21) | (hour <= 5)).astype(np.int8)
    # calendar
    month = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["dom"] = dom
    X["is_month_end"] = (dom >= 28).astype(np.int8)
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)
    # categoricals (unseen levels -> NaN)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def make_model(n_estimators: int, **kw) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        learning_rate=0.01,
        max_depth=16,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.7,
        tree_method="hist",
        max_bin=512,
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


t0 = time.time()
# fixed tree count: ES on train-internal splits picks too many trees for the 2006 shift
model = make_model(1500)
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
