"""XGBoost binary classifier for airline delay prediction.

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

# --- feature engineering -------------------------------------------------------
NUM_COLS = ["DepTime", "Distance"]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def _num(s):
    """Extract the numeric part of strings like 'c-4' -> 4. Missing -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _hour(deptime):
    """CRS DepTime (hhmm int) -> fractional hour of day."""
    t = pd.to_numeric(deptime, errors="coerce").astype(float)
    hh = np.floor(t / 100.0)
    mm = t - hh * 100.0
    bad = (t < 0) | (t > 2400) | (mm > 59)
    out = np.where(bad, np.nan, hh + mm / 60.0)
    return pd.Series(out, index=deptime.index)


# Per-key smoothed historical delay-rate statistics, computed on TRAIN rows only.
_train_hour = _hour(train["DepTime"])
_train_route = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
_train_oh = train["Origin"].astype(str) + "-" + _train_hour.round().fillna(-1).astype(int).astype(str)
_prior = float((train[TARGET] == POSITIVE).mean())


def _make_stat(key_series_train, y_train_int, sigma):
    """Empirical-Bayes smoothed target rate per key, fit on train rows only."""
    g = y_train_int.groupby(key_series_train)
    stats = pd.DataFrame({"mean": g.mean(), "count": g.size()})
    stats["te"] = (stats["mean"] * stats["count"] + _prior * sigma) / (stats["count"] + sigma)
    return stats


STATS = {}
_yt = (train[TARGET] == POSITIVE).astype(int)
STATS["Origin"] = _make_stat(train["Origin"].astype(str), _yt, 7.0)
STATS["Dest"] = _make_stat(train["Dest"].astype(str), _yt, 7.0)
STATS["UniqueCarrier"] = _make_stat(train["UniqueCarrier"].astype(str), _yt, 7.0)
STATS["Route"] = _make_stat(_train_route, _yt, 7.0)
STATS["Origin-hour"] = _make_stat(_train_oh, _yt, 20.0)
# note: Route/Origin-hour stats are keyed by the exact same strings prepare() builds.


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    hr = _hour(df["DepTime"])
    mon = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    X["hour"] = hr
    X["sin_h"] = np.sin(2 * np.pi * hr / 24.0)
    X["cos_h"] = np.cos(2 * np.pi * hr / 24.0)
    X["sin_m"] = np.sin(2 * np.pi * mon / 12.0)
    X["cos_m"] = np.cos(2 * np.pi * mon / 12.0)
    # categorical codes for the tree
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    # target statistics (fit on train only; unseen keys -> NaN)
    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    oh = df["Origin"].astype(str) + "-" + hr.round().fillna(-1).astype(int).astype(str)
    X["te_origin"] = STATS["Origin"]["te"].reindex(df["Origin"].astype(str)).to_numpy()
    X["te_dest"] = STATS["Dest"]["te"].reindex(df["Dest"].astype(str)).to_numpy()
    X["te_carrier"] = STATS["UniqueCarrier"]["te"].reindex(df["UniqueCarrier"].astype(str)).to_numpy()
    X["te_route"] = STATS["Route"]["te"].reindex(route).to_numpy()
    X["te_oh"] = STATS["Origin-hour"]["te"].reindex(oh).to_numpy()
    return X


CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train),
    to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s, best iters: {model.best_iteration}")

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
