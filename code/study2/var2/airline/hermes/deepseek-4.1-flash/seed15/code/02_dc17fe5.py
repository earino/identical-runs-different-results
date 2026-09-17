"""XGBoost binary classifier for flight-departure-delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes
------------
The eval slice is a *different year* than train, so anything the model can only learn as a
year-specific memorised pattern actively hurts. Two consequences drive this file:
  * high-cardinality categorical splits (Origin/Dest/Route as native categoricals) overfit badly,
    so those columns are converted to target statistics instead;
  * the target statistics are means of the TRAIN labels only, smoothed towards the global prior,
    and the train matrix itself uses out-of-fold values so the model cannot memorise them.
All feature engineering lives in `add_features()`/`prepare()`, which are pure functions of the raw
frame plus lookup tables fitted on train, so they reproduce exactly on the hidden holdout.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

_MONTH_DAYS = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_CUM_DAYS = np.cumsum(_MONTH_DAYS)  # _CUM_DAYS[m-1] = number of days before month m
TE_K = 20      # target-encoding smoothing strength (pseudo-counts towards the prior)
TE_FOLDS = 5   # folds used to build out-of-fold encodings for the training matrix

# keys that get a smoothed target statistic; all are cheap, year-stable groupings
TE_KEYS = [
    "org", "dst", "route", "carrier",
    "org_hour", "org_tod", "carrier_tod", "route_hour",
    "dst_hour", "carrier_hour", "route_tod", "org_tod30", "carrier_tod30",
]
# keys that get a (label-free) share-of-flights feature
FR_KEYS = ["route", "org", "carrier", "dst", "route_hour", "org_tod", "route_tod"]
NUM_COLS = ["month", "dom", "dow", "hour", "minute", "tod", "distance"]
TE_COLS = ["te_" + c for c in TE_KEYS]
FR_COLS = ["f_" + c for c in FR_KEYS]
FEATURES = NUM_COLS + TE_COLS + FR_COLS


def _num(series: pd.Series) -> np.ndarray:
    """'c-7' -> 7."""
    return series.astype(str).str[2:].astype(int).to_numpy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free features + the raw keys used for target/share encodings."""
    out = pd.DataFrame(index=df.index)
    month = _num(df["Month"])
    out["month"] = month
    out["dom"] = _num(df["DayofMonth"])
    out["dow"] = _num(df["DayOfWeek"])

    dt = df["DepTime"].to_numpy(dtype=np.int64)
    hour = (dt // 100) % 24           # 2400-2620 roll over into the next day
    minute = dt % 100
    out["hour"] = hour
    out["minute"] = minute
    out["tod"] = hour * 60 + minute
    out["tod30"] = out["tod"] // 30
    out["distance"] = df["Distance"].to_numpy(dtype=np.float64)

    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    route = org + "_" + dst
    out["org"] = org
    out["dst"] = dst
    out["carrier"] = carrier
    out["route"] = route
    for a, b in [("org", "hour"), ("org", "tod"), ("dst", "hour"), ("carrier", "hour"),
                 ("carrier", "tod"), ("route", "hour"), ("route", "tod"),
                 ("org", "tod30"), ("carrier", "tod30")]:
        out[a + "_" + b] = out[a] + "_" + out[b].astype(str)
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- encoders fitted on the TRAINING labels only -------------------------------
y_train = to_y(train)
_prior = float(y_train.mean())
_train_feats = add_features(train)


def _target_stats(keys: pd.Series, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + _prior * TE_K) / (g["count"] + TE_K)


TE_MAPS = {c: _target_stats(_train_feats[c], y_train) for c in TE_KEYS}

# out-of-fold encodings so the model does not see its own row's label through a target statistic
OOF = {}
for c in TE_KEYS:
    keys = _train_feats[c].to_numpy()
    enc = np.full(len(keys), _prior, dtype=np.float64)
    for fit_idx, val_idx in KFold(TE_FOLDS, shuffle=True, random_state=SEED).split(keys):
        m = _target_stats(pd.Series(keys[fit_idx]), y_train[fit_idx])
        enc[val_idx] = pd.Series(keys[val_idx]).map(m).fillna(_prior).to_numpy()
    OOF[c] = enc


def prepare(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = add_features(df)
    for c in TE_KEYS:
        if oof:
            X["te_" + c] = OOF[c]      # only valid for the training frame, in its original row order
        else:
            X["te_" + c] = X[c].map(TE_MAPS[c]).fillna(_prior).to_numpy()
    n = max(len(X), 1)
    for c in FR_KEYS:
        X["f_" + c] = (X[c].map(X[c].value_counts()) / n).to_numpy()
    return X[FEATURES]


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    reg_lambda=10,
    tree_method="hist",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train, oof=True), y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
