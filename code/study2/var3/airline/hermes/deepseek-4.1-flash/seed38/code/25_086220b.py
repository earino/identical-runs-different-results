"""XGBoost binary classifier for flight-delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# columns whose values are string codes of the form "c-<int>"
CODE_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _decode(series: pd.Series) -> pd.Series:
    """'c-11' -> 11.0 (NaN when the value is missing or malformed)."""
    return pd.to_numeric(series.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CODE_COLS:
        X[c] = _decode(df[c])
    # DepTime is scheduled departure as hhmm; decompose into time-of-day position (cycle-safe sin/cos).
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(lower=0, upper=28)
    minute = dep % 100
    tod = (hour * 60 + minute).clip(lower=0, upper=28 * 60 + 59)
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    # Distance: raw + log, plus a coarse flight-length band.
    dist = pd.to_numeric(df["Distance"], errors="coerce").clip(lower=1)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=12,
    eta=0.02,
    subsample=0.6,
    colsample_bytree=0.6,
    min_child_weight=10,
    reg_lambda=5.0,
    tree_method="hist",
    max_cat_to_onehot=4,
    max_cat_threshold=256,
    seed=SEED,
    nthread=N_JOBS,
)


def _dm(X: pd.DataFrame, y: np.ndarray = None) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def _fit(params, X, y, rounds, Xv=None, yv=None):
    if Xv is None:
        return xgb.train(params, _dm(X, y), num_boost_round=rounds)
    es = xgb.callback.EarlyStopping(rounds=50, save_best=False)
    return xgb.train(params, _dm(X, y), num_boost_round=rounds,
                     evals=[(_dm(Xv, yv), "valid")], callbacks=[es], verbose_eval=False)


CONFIGS = [(dict(PARAMS), 6)]
ROUND_CAP = 1400   # the experiment is killed at 120s, so keep the whole run bounded
SEEDS = [SEED + 101 * i for i in range(6)]

t0 = time.time()
Xall, yall = prepare(train), to_y(train)
# hold out 15% of train to pick the boosting round of each member, then refit on all of it
rs = np.random.RandomState(SEED)
vi = rs.rand(len(train)) < 0.15
members = []
for cfg, n_seeds in CONFIGS:
    probe = _fit(cfg, Xall[~vi], yall[~vi], 3000, Xall[vi], yall[vi])
    # the internal 2005 validation optimum tends to overshoot the 2006 optimum, so shrink it a little
    rounds = min(int(0.30 * int(probe.best_iteration)) + 1, ROUND_CAP)
    print(f"probe depth={cfg['max_depth']} eta={cfg['eta']} best_iteration={probe.best_iteration} -> {rounds} rounds")
    for s in SEEDS[:n_seeds]:
        members.append((_fit(dict(cfg, seed=s), Xall, yall, rounds), rounds))
print(f"Training time: {time.time() - t0:.1f}s ({len(members)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    d = _dm(prepare(df))
    return np.mean([b.predict(d, iteration_range=(0, r)) for b, r in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
