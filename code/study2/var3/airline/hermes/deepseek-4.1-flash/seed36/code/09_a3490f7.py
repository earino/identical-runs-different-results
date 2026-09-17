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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
C_ORD = ["Month", "DayofMonth", "DayOfWeek"]          # encoded as "c-<n>" strings -> ordinal ints
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]        # native xgboost categoricals
RAW_COLS = C_ORD + ["DepTime"] + CAT_COLS + ["Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _ordinal(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _clock(dep: pd.Series):
    t = pd.to_numeric(dep, errors="coerce").astype(float)
    t = t.where(t < 2400, t - 2400)                   # 24xx-26xx are after-midnight departures
    h = np.floor(t / 100.0)
    return h, h * 60.0 + (t - h * 100.0)              # hour, minutes since midnight


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical keys whose frequency/congestion counts are computed on the training split only."""
    hour = _clock(df["DepTime"])[0].fillna(-1).astype(int).astype(str)
    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    return pd.DataFrame(
        {
            "Origin": org,
            "Dest": dst,
            "UniqueCarrier": df["UniqueCarrier"].astype(str),
            "route": org + "_" + dst,
            "orig_hour": org + "_" + hour,
            "dest_hour": dst + "_" + hour,
        },
        index=df.index,
    )


# fitted on the TRAINING split only (never on the dataframe passed to prepare)
FREQ_MAPS = {c: _keys(train)[c].value_counts() for c in _keys(train).columns}

# --- smoothed target encoding (train-only; out-of-fold values for the training rows) ------------
TE_ALPHA = 60.0


def _te_maps(keys: pd.DataFrame, y: np.ndarray, prior: float) -> dict:
    out = {}
    for c in keys.columns:
        g = pd.DataFrame({"k": keys[c].to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        out[c] = (g["sum"] + prior * TE_ALPHA) / (g["count"] + TE_ALPHA)
    return out


def _te_oof(keys: pd.DataFrame, y: np.ndarray, prior: float, n_splits: int = 5) -> pd.DataFrame:
    from sklearn.model_selection import KFold

    vals = {c: np.full(len(keys), prior, dtype=float) for c in keys.columns}
    for tr_idx, va_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(keys):
        m = _te_maps(keys.iloc[tr_idx], y[tr_idx], prior)
        for c in keys.columns:
            vals[c][va_idx] = keys[c].iloc[va_idx].map(m[c]).to_numpy(dtype=float)
    return pd.DataFrame({c: np.nan_to_num(v, nan=prior) for c, v in vals.items()}, index=keys.index)


_KEYS_TRAIN = _keys(train)
_Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(_Y_TRAIN.mean())
TE_MAPS = _te_maps(_KEYS_TRAIN, _Y_TRAIN, PRIOR)        # full-train maps, used at inference time
OOF_TE = _te_oof(_KEYS_TRAIN, _Y_TRAIN, PRIOR)          # leakage-free encodings for fitting


# --- hourly schedule profile of each airport (train-only): a scale-free congestion signal ---------
def _hourly(key_col: str) -> pd.Series:
    h = _clock(train["DepTime"])[0].fillna(-1).astype(int)
    g = pd.DataFrame({"k": train[key_col].astype(str).to_numpy(), "h": h.to_numpy()})
    cnt = g.groupby(["k", "h"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    share = cnt.div(cnt.sum(axis=1), axis=0)
    wide = share + share.shift(1, axis=1).fillna(0.0) + share.shift(-1, axis=1).fillna(0.0)
    return share.stack(), wide.stack()                  # MultiIndex (airport, hour) -> share


ORIG_SHARE, ORIG_WIDE = _hourly("Origin")
DEST_SHARE, DEST_WIDE = _hourly("Dest")


def _at(stacked: pd.Series, idx: pd.Series, h: pd.Series) -> np.ndarray:
    mi = pd.MultiIndex.from_arrays([idx.to_numpy(), h.to_numpy()])
    return np.nan_to_num(stacked.reindex(mi).to_numpy(dtype=float), nan=0.0)


def prepare(df: pd.DataFrame, te: pd.DataFrame | None = None) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in C_ORD:
        X[c] = _ordinal(df[c])
    hour, mins = _clock(df["DepTime"])
    X["dep_hour"] = hour
    X["dep_min"] = mins
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    hh = hour.fillna(-1).astype(int)
    X["share_org_hour"] = _at(ORIG_SHARE, org, hh)
    X["share_dst_hour"] = _at(DEST_SHARE, dst, hh)
    X["share_org_3h"] = _at(ORIG_WIDE, org, hh)
    X["share_dst_3h"] = _at(DEST_WIDE, dst, hh)
    ks = _keys(df)
    for c, vc in FREQ_MAPS.items():
        X["cnt_" + c] = np.log1p(ks[c].map(vc).fillna(0.0).to_numpy(dtype=float))
    if te is None:
        for c, m in TE_MAPS.items():
            X["te_" + c] = np.nan_to_num(ks[c].map(m).to_numpy(dtype=float), nan=PRIOR)
    else:
        for c in TE_MAPS:
            X["te_" + c] = te[c].to_numpy(dtype=float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])   # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    max_bin=512,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
MEMBERS = [(s, d) for d in (5, 6, 7) for s in (42, 7, 2024)]  # (seed, max_depth): decorrelate by capacity

t0 = time.time()
X_train = prepare(train, te=OOF_TE)
y_train = to_y(train)
models = [
    xgb.XGBClassifier(random_state=s, **{**PARAMS, "max_depth": d}).fit(X_train, y_train)
    for s, d in MEMBERS
]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
