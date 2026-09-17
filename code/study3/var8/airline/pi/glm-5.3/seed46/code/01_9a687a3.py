"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

v3 (findings from exploration):
  - The 2005->2006 time shift punishes capacity tuned on 2005 data; regularized DEEP trees with few
    boosting rounds generalize best (shallow/many-round variants overfit 2005 specifics).
  - Useful features: parsed Month/DayOfMonth/DayOfWeek ints, DepTime -> hour/minute/frac, hour as a
    native categorical, log route/origin/dest flight frequencies (year-stable), carrier/origin/dest cats.
  - Route pair categorical and target encodings HURT (2005-specific rates don't transfer).
  - Small diverse ensemble: 6 configs (deep trees, subsample 0.7-0.75, 150 rounds), mean of margins.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# All feature engineering lives inside prepare(); predict_proba() calls it on unseen rows.
CATS = ["UniqueCarrier", "Origin", "Dest"]
HOUR_LEVELS = [str(i) for i in range(24)]
# encoder state fitted on the 2005 training data ONLY (never on the frame passed to predict_proba)
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CATS}
freq_maps = {
    "route": train.groupby(train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).size(),
    "origin": train.groupby(train["Origin"].astype(str)).size(),
    "dest": train.groupby(train["Dest"].astype(str)).size(),
}
MED_DEPTIME = float(train["DepTime"].median())
MED_DIST = float(train["Distance"].median())


def _freq(col: pd.Series, mapping: pd.Series) -> np.ndarray:
    s = pd.Series(col.astype(str).to_numpy())
    return np.log1p(s.map(mapping).fillna(0.0).to_numpy(dtype=float))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        X[c] = df[c].astype(str).str.slice(2).astype(float)
    dep = df["DepTime"].astype(float).fillna(MED_DEPTIME)
    hh = np.floor(dep / 100.0) % 24.0
    mm = dep - np.floor(dep / 100.0) * 100.0
    X["DepHour"] = hh
    X["DepMin"] = mm
    X["DepTimeFrac"] = hh + mm / 60.0
    X["DepHourC"] = pd.Categorical(hh.astype(int).astype(str), categories=HOUR_LEVELS)
    X["Distance"] = df["Distance"].astype(float).fillna(MED_DIST)
    X["freq_route"] = _freq(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), freq_maps["route"])
    X["freq_origin"] = _freq(df["Origin"], freq_maps["origin"])
    X["freq_dest"] = _freq(df["Dest"], freq_maps["dest"])
    for c in CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small diverse ensemble of regularized deep XGBoost models ----------
BASE = dict(
    objective="binary:logistic",
    tree_method="hist",
    nthread=N_JOBS,
    eta=0.07,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=10.0,
)
CONFIGS = [  # (max_depth, subsample)
    (16, 0.70),
    (18, 0.70),
    (20, 0.70),
    (20, 0.75),
    (22, 0.70),
    (24, 0.70),
]
ROUNDS = 150

t0 = time.time()
dtrain = xgb.DMatrix(prepare(train), label=to_y(train), enable_categorical=True)
models = [
    xgb.train({**BASE, "max_depth": d, "subsample": s, "seed": i}, dtrain, num_boost_round=ROUNDS)
    for i, (d, s) in enumerate(CONFIGS, start=1)
]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dmat = xgb.DMatrix(prepare(df), enable_categorical=True)
    margin = np.mean([m.predict(dmat, output_margin=True) for m in models], axis=0)
    return 1.0 / (1.0 + np.exp(-margin))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
