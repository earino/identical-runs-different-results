"""XGBoost binary classifier for flight-departure-delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes
------------
The eval slice is a *different year* than train, so the score measures transfer, not fit. Everything here
follows from that:
  * native categorical splits on Origin/Dest/Route (hundreds-thousands of levels) overfit badly and are
    replaced by smooth statistics;
  * every interesting key (route, carrier, airport, airport x hour, route x hour, day x month, ...) gives
    - a smoothed *label* statistic: mean of `dep_delayed_15min` over training rows with that key. Built from
      the training labels only, and the training matrix gets out-of-fold values so the model cannot
      memorise its own rows;
    - a label-free *traffic statistic* computed from the scored frame:
        s_<key>  = share of flights in the frame with that key (route/airport size predicts delays),
        sc_<key> = the same share *conditional on a coarser parent* (e.g. this route's share among flights
                   in the same hour). The conditional form is used for keys whose raw share depends on the
                   frame's month mix, so the feature means the same thing whether the frame is a full year
                   or a season.
All of this lives in `add_features()`/`prepare()`, which are pure functions of the raw frame plus lookup
tables fitted on train, so they reproduce exactly on the hidden holdout.
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
TE_K = 20      # smoothing (pseudo-counts towards the prior) for target statistics
TE_FOLDS = 5   # folds used to build out-of-fold encodings for the training matrix

# keys that get a smoothed target statistic (label-based)
TE_KEYS = [
    "org", "dst", "route", "carrier",
    "org_hour", "org_tod", "carrier_tod", "route_hour",
    "dst_hour", "carrier_hour", "route_tod", "org_tod30", "carrier_tod30",
]
# key pairs that get a marginal traffic share (label-free)
_PAIRS = [
    ("org", "hour"), ("org", "tod"), ("org", "tod30"), ("org", "tod10"), ("org", "dow"),
    ("org", "dom"), ("org", "month"),
    ("dst", "hour"), ("dst", "tod"), ("dst", "tod30"), ("dst", "tod10"), ("dst", "dow"), ("dst", "month"),
    ("carrier", "hour"), ("carrier", "tod"), ("carrier", "tod30"), ("carrier", "tod10"), ("carrier", "dow"),
    ("route", "hour"), ("route", "tod"), ("route", "tod30"), ("route", "tod10"), ("route", "dow"),
    ("route", "month"), ("route", "carrier"), ("route", "dom"),
    ("hour", "dow"), ("tod", "dow"), ("tod10", "dow"), ("tod30", "dow"),
    ("dom", "hour"), ("dom", "carrier"), ("dom", "dow"), ("dom", "month"),
    ("month", "carrier"), ("month", "org"),
    ("dist_bin", "carrier"),
]
SHARE_KEYS = ["org", "dst", "route", "carrier"] + [f"{a}_{b}" for a, b in _PAIRS]
# (parent, child): child's share among rows sharing the parent -> composition-invariant
_COND = [("month", "dom_month"), ("hour", "route_hour"), ("month", "route_month")]
COND_KEYS = [c for _, c in _COND]

NUM_COLS = ["month", "dom", "dow", "hour", "minute", "tod", "distance"]
TE_COLS = ["te_" + c for c in TE_KEYS]
SHARE_COLS = ["s_" + c for c in SHARE_KEYS]
COND_COLS = ["sc_" + c for c in COND_KEYS]
FEATURES = NUM_COLS + TE_COLS + SHARE_COLS + COND_COLS


def _num(series: pd.Series) -> np.ndarray:
    """'c-7' -> 7."""
    return series.astype(str).str[2:].astype(int).to_numpy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free numeric features plus the raw keys used for target/traffic encodings."""
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
    out["tod10"] = out["tod"] // 10
    distance = df["Distance"].to_numpy(dtype=np.float64)
    out["distance"] = distance
    out["dist_bin"] = (np.log1p(distance) * 3).astype(int)

    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    out["org"] = org
    out["dst"] = dst
    out["carrier"] = df["UniqueCarrier"].astype(str)
    out["route"] = org + "_" + dst
    for a, b in _PAIRS:
        out[a + "_" + b] = out[a].astype(str) + "_" + out[b].astype(str)
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- label statistics and traffic shares, fitted on the TRAINING data only -----
y_train = to_y(train)
_prior = float(y_train.mean())
_train_feats = add_features(train)


def _target_stats(keys: pd.Series, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + _prior * TE_K) / (g["count"] + TE_K)


TE_MAPS = {c: _target_stats(_train_feats[c], y_train) for c in TE_KEYS}

# out-of-fold label statistics, so no training row sees an encoding built from its own label
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
    for c in SHARE_KEYS:
        X["s_" + c] = (X[c].map(X[c].value_counts()) / n).to_numpy()
    for parent, child in _COND:        # child's share within the parent group
        X["sc_" + child] = X.groupby(parent)[child].transform(
            lambda s: s.map(s.value_counts()) / max(len(s), 1)).to_numpy()
    return X[FEATURES]


# --- model --------------------------------------------------------------------
# A small seed ensemble: the individual models are cheap (~5 s each) and averaging them removes the
# seed-to-seed wobble without changing the model class.
SEEDS = (42, 7, 2024)
MODEL_PARAMS = dict(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    tree_method="hist",
    n_jobs=N_JOBS,
)

X_fit = prepare(train, oof=True)
t0 = time.time()
models = [xgb.XGBClassifier(random_state=s, **MODEL_PARAMS).fit(X_fit, y_train) for s in SEEDS]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
