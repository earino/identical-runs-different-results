"""XGBoost binary classifier: airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _tod_minutes(s: pd.Series) -> pd.Series:
    t = pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")
    return (t // 100) * 60 + (t % 100)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    ang = 2 * np.pi * _tod_minutes(df["DepTime"]) / 1440.0
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["tod_min"] = _tod_minutes(df["DepTime"])
    X["month"] = pd.to_numeric(df["Month"].str[2:], errors="coerce").fillna(0).astype("int64")
    X["day"] = pd.to_numeric(df["DayofMonth"].str[2:], errors="coerce").fillna(0).astype("int64")
    X["dow"] = pd.to_numeric(df["DayOfWeek"].str[2:], errors="coerce").fillna(0).astype("int64")
    X["distance"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce").fillna(0))
    mins = _tod_minutes(df["DepTime"])
    X["hour"] = (mins // 60).astype("int64")
    dow = pd.to_numeric(df["DayOfWeek"].str[2:], errors="coerce").fillna(0).astype("int64")
    car = df["UniqueCarrier"].astype(str)
    X["car_hour"] = pd.Categorical(car + "_" + (mins // 60).astype(str),
                                   categories=sorted((train["UniqueCarrier"].astype(str) + "_" +
                                                      (_tod_minutes(train["DepTime"]) // 60).astype(str)).unique()))
    X["dow_hour"] = pd.Categorical(dow.astype(str) + "_" + (mins // 60).astype(str),
                                   categories=sorted((pd.to_numeric(train["DayOfWeek"].str[2:], errors="coerce").fillna(0).astype("int64").astype(str) + "_" +
                                                      (_tod_minutes(train["DepTime"]) // 60).astype(str)).unique()))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 1337, 2024, 7, 99]

t0 = time.time()
members = []
for seed in SEEDS:
    rng = np.random.RandomState(seed)
    tr_idx = np.arange(len(train))
    rng.shuffle(tr_idx)
    cut = int(0.85 * len(tr_idx))
    fit_idx, val_idx = tr_idx[:cut], tr_idx[cut:]
    es = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=50, random_state=seed, **PARAMS)
    es.fit(
        prepare(train.iloc[fit_idx]), to_y(train.iloc[fit_idx]),
        eval_set=[(prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx]))],
        verbose=0,
    )
    ntree = es.best_iteration + 1
    m = xgb.XGBClassifier(n_estimators=ntree, random_state=seed, **PARAMS)
    m.fit(prepare(train), to_y(train))
    members.append(m)
    print(f"seed {seed}: ntree {ntree}, ES time {time.time() - t0:.1f}s")
print(f"Ensemble of {len(members)} trained in {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
