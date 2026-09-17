"""XGBoost binary classifier for airline departure-delay prediction (dep_delayed_15min).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare()/prepare_te(), so it transfers to unseen rows.
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

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# fitted on TRAIN ONLY, then reused verbatim when preparing any unseen dataframe
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df):
    return df["DepTime"].to_numpy() // 100


def _carrier_hour(df):
    return df["UniqueCarrier"].astype(str) + "_" + _hour(df).astype(str)


ch_levels = pd.Index(sorted(_carrier_hour(train).unique()))


def _ordinal(df, col):
    """c-<n> string level -> integer n."""
    return df[col].astype(str).str.slice(2).astype(int).to_numpy()


def _time_block(df):
    """numeric time features shared by both feature views."""
    d = df["DepTime"].to_numpy()
    h, m = d // 100, d % 100
    tod = h * 60 + m
    dow = _ordinal(df, "DayOfWeek")
    out = pd.DataFrame(index=df.index)
    out["DepTime"] = d
    out["Distance"] = df["Distance"].to_numpy()
    out["dep_hour"] = h
    out["dep_min"] = m
    out["dep_tod"] = tod
    out["sin_h"] = np.sin(2 * np.pi * h / 24.0)
    out["cos_h"] = np.cos(2 * np.pi * h / 24.0)
    out["sin_t"] = np.sin(2 * np.pi * tod / 1440.0)
    out["cos_t"] = np.cos(2 * np.pi * tod / 1440.0)
    out["dow"] = dow
    out["mon"] = _ordinal(df, "Month")
    out["dom"] = _ordinal(df, "DayofMonth")
    out["is_weekend"] = (dow >= 6).astype(int)
    out["logdist"] = np.log1p(out["Distance"])
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """View 1: native categorical handling (XGBoost partition splits)."""
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["carrier_hour"] = pd.Categorical(_carrier_hour(df), categories=ch_levels)
    return pd.concat([X, _time_block(df)], axis=1)


# --- view 2: out-of-fold target encoding of the high-cardinality columns --------
TE_SPECS = {
    "carrier": ["UniqueCarrier"],
    "origin": ["Origin"],
    "dest": ["Dest"],
    "route": ["Origin", "Dest"],
    "carrier_hour": ["UniqueCarrier", "hour"],
    "carrier_origin": ["UniqueCarrier", "Origin"],
    "origin_hour": ["Origin", "hour"],
    "carrier_dow": ["UniqueCarrier", "dow"],
    "dest_hour": ["Dest", "hour"],
    "route_dow": ["Origin", "Dest", "dow"],
    "carrier_route": ["UniqueCarrier", "Origin", "Dest"],
}
TE_NAMES = list(TE_SPECS)
TE_K = 100.0


def _te_part(df, col):
    if col == "hour":
        return pd.Series(_hour(df).astype(str), index=df.index)
    if col == "dow":
        return pd.Series(_ordinal(df, "DayOfWeek").astype(str), index=df.index)
    return df[col].astype(str)


def _te_key(df, name):
    parts = [_te_part(df, c) for c in TE_SPECS[name]]
    if len(parts) == 1:
        return parts[0]
    return parts[0].str.cat(parts[1:], sep="|")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PRIOR = float(to_y(train).mean())
_y_train = to_y(train)


def _te_map(keys, yv):
    g = pd.DataFrame({"k": keys, "y": yv}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * TE_K) / (g["count"] + TE_K)


TE_KEYS = {n: _te_key(train, n).to_numpy() for n in TE_NAMES}
TE_MAPS = {n: _te_map(TE_KEYS[n], _y_train) for n in TE_NAMES}


def _oof_te(name, nfold=5):
    keys = TE_KEYS[name]
    out = np.full(len(keys), PRIOR)
    for tri, vai in KFold(nfold, shuffle=True, random_state=0).split(keys):
        m = _te_map(keys[tri], _y_train[tri])
        out[vai] = pd.Series(keys[vai]).map(m).fillna(PRIOR).to_numpy()
    return out


OOF_TE = {n: _oof_te(n) for n in TE_NAMES}


def prepare_te(df: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    """View 2: numeric features + smoothed target encodings (train-only statistics)."""
    X = _time_block(df)
    for n in TE_NAMES:
        X["te_" + n] = OOF_TE[n] if oof else _te_key(df, n).map(TE_MAPS[n]).fillna(PRIOR).to_numpy()
    return X


# --- model --------------------------------------------------------------------
# Ensemble of XGBoost models with deliberately different tree structures and feature views;
# averaging them is more robust on unseen (later-year) rows than any single configuration.
BASE = dict(
    n_estimators=500,
    learning_rate=0.04,
    tree_method="hist",
    enable_categorical=True,
    colsample_bylevel=0.8,
    reg_lambda=3.0,
    n_jobs=N_JOBS,
)
PARAMS = [
    dict(max_depth=8, colsample_bylevel=0.7, reg_alpha=1.0),
    dict(max_depth=9, colsample_bylevel=0.65, min_child_weight=3),
    dict(max_depth=4, colsample_bylevel=0.6, n_estimators=1400, learning_rate=0.025),
    dict(max_depth=8, colsample_bylevel=0.7, reg_alpha=1.0),         # second seed
    dict(max_depth=10, colsample_bylevel=0.6, min_child_weight=5),
    dict(max_depth=9, colsample_bylevel=0.65, min_child_weight=3),  # second seed
]
SEEDS = [42, 43, 44, 142, 45, 243]
# one extra member on the target-encoded view, for encoding diversity
TE_PARAMS = [
    dict(max_depth=8, colsample_bylevel=0.7, reg_alpha=1.0),
    dict(max_depth=10, colsample_bylevel=0.6, min_child_weight=5),
    dict(max_depth=4, colsample_bylevel=0.6, n_estimators=1400, learning_rate=0.025),
    dict(max_depth=9, colsample_bylevel=0.65, min_child_weight=3),
    dict(max_depth=7, colsample_bylevel=0.75),
]
TE_SEEDS = [142, 46, 47, 48, 49]


def build_models():
    """-> list of (model, prepare_fn) pairs."""
    out = []
    for i, p in enumerate(PARAMS):
        kw = dict(BASE)
        kw.update(p)
        kw["random_state"] = SEEDS[i]
        out.append((xgb.XGBClassifier(**kw), prepare))
    for i, p in enumerate(TE_PARAMS):
        kw = dict(BASE)
        kw.update(p)
        kw["random_state"] = TE_SEEDS[i]
        out.append((xgb.XGBClassifier(**kw), "te"))
    return out


models = build_models()

t0 = time.time()
for m, view in models:
    m.fit(prepare_te(train, oof=True) if view == "te" else prepare(train), _y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    views = {}
    preds = []
    for m, view in models:
        if view not in views:
            views[view] = prepare_te(df) if view == "te" else prepare(df)
        preds.append(m.predict_proba(views[view])[:, 1])
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
