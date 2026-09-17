"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Notes from experiments so far:
  - Early stopping on a 2005 random split picks ~10x too many rounds for 2006 generalization
    (2005-val AUC keeps rising while 2006 eval AUC peaks at 100-300 rounds then decays). Fixed rounds.
  - route (Origin x Dest) categorical overfits 2005 and hurts 2006 transfer. Dropped.
  - Time-of-day features from DepTime are the strongest signal.
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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
DIST_Q = train["Distance"].quantile([1 / 3, 2 / 3]).to_numpy()  # tercile edges from training data

# target-encoding keys: raw column name -> function extracting the key from a row
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "hour", "dow", "hour_carrier", "hour_origin",
           "hour_dest", "dow_carrier", "dow_origin", "route", "hour_dow", "arr_hour",
           "arr_carrier", "arr_dest", "tod48", "tod48_carrier", "tod96", "tod96_carrier",
           "arr48", "arr48_carrier", "arr96", "arr96_carrier", "tod192", "tod192_carrier",
           "tod48_dow", "dest_tod6", "origin_dist3", "tod48_dist3"]
TE_M = {"UniqueCarrier": 50, "Origin": 50, "Dest": 50, "hour": 50, "dow": 50,
        "hour_carrier": 100, "hour_origin": 200, "hour_dest": 200,
        "dow_carrier": 100, "dow_origin": 200, "route": 300, "hour_dow": 100,
        "arr_hour": 50, "arr_carrier": 100, "arr_dest": 200,
        "tod48": 50, "tod48_carrier": 100, "tod96": 100, "tod96_carrier": 150,
        "arr48": 50, "arr48_carrier": 100, "arr96": 100, "arr96_carrier": 150,
        "tod192": 100, "tod192_carrier": 200,
        "tod48_dow": 100, "dest_tod6": 150,
        "origin_dist3": 150, "tod48_dist3": 100}


def _num(s: pd.Series) -> pd.Series:
    """c-7 -> 7 (Month/DayofMonth/DayOfWeek arrive as c-<n> strings)."""
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def _te_key(df: pd.DataFrame) -> dict:
    """Extract the (categorical) key series TE is computed on, from raw columns."""
    dep = df["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 24)
    tod = hour + (dep - hour * 100) / 60.0
    arr_tod = (tod + df["Distance"].astype(float) / 500.0 + 0.5) % 24.0
    tod48 = (tod * 2).astype(int).clip(0, 47)
    tod96 = (tod * 4).astype(int).clip(0, 95)
    tod192 = (tod * 8).astype(int).clip(0, 191)
    dist3 = np.digitize(df["Distance"].astype(float), DIST_Q)
    arr48 = (arr_tod * 2).astype(int).clip(0, 47)
    arr96 = (arr_tod * 4).astype(int).clip(0, 95)
    return {
        "UniqueCarrier": df["UniqueCarrier"].astype(str),
        "Origin": df["Origin"].astype(str),
        "Dest": df["Dest"].astype(str),
        "hour": hour,
        "dow": df["DayOfWeek"].astype(str),
        "hour_carrier": hour.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "hour_origin": hour.astype(str) + "_" + df["Origin"].astype(str),
        "hour_dest": hour.astype(str) + "_" + df["Dest"].astype(str),
        "dow_carrier": df["DayOfWeek"].astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "dow_origin": df["DayOfWeek"].astype(str) + "_" + df["Origin"].astype(str),
        "route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
        "hour_dow": hour.astype(str) + "_" + df["DayOfWeek"].astype(str),
        "arr_hour": arr_tod.astype(int).clip(0, 23),
        "arr_carrier": arr_tod.astype(int).clip(0, 23).astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "arr_dest": arr_tod.astype(int).clip(0, 23).astype(str) + "_" + df["Dest"].astype(str),
        "tod48": tod48,
        "tod48_carrier": tod48.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "tod96": tod96,
        "tod96_carrier": tod96.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "arr48": arr48,
        "arr48_carrier": arr48.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "arr96": arr96,
        "arr96_carrier": arr96.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "tod192": tod192,
        "tod192_carrier": tod192.astype(str) + "_" + df["UniqueCarrier"].astype(str),
        "tod48_dow": tod48.astype(str) + "_" + df["DayOfWeek"].astype(str),
        "dest_tod6": df["Dest"].astype(str) + "_" + (tod // 4).astype(int).clip(0, 5).astype(str),
        "origin_dist3": df["Origin"].astype(str) + "_" + pd.Series(dist3, index=df.index).astype(str),
        "tod48_dist3": tod48.astype(str) + "_" + pd.Series(dist3, index=df.index).astype(str),
    }


def _te_map(keys: pd.Series, y: np.ndarray, m: int) -> pd.Series:
    """Smoothed target mean per key, fit on given rows only."""
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * y.mean()) / (g["count"] + m)


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everything except the TE columns (identical for train rows and unseen rows)."""
    mon = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    dep = df["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 24)
    tod = hour + (dep - hour * 100) / 60.0  # time of day in hours
    doy = (mon - 1) * 30.5 + dom  # rough position in year

    X = pd.DataFrame(index=df.index)
    X["DepTime"] = dep
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["Month"] = mon
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["Distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    # estimated arrival time: cruise ~500 mph plus ~30 min taxi/climb/descent
    arr_tod = (tod + (df["Distance"].astype(float) / 500.0 + 0.5)) % 24.0
    X["arr_tod"] = arr_tod
    X["arr_tod_sin"] = np.sin(2 * np.pi * arr_tod / 24.0)
    X["arr_tod_cos"] = np.cos(2 * np.pi * arr_tod / 24.0)
    X["arr_overnight"] = (arr_tod < tod).astype(int)
    return X


def _add_cat(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].values, categories=cat_levels[c])
    return X


def _add_te(X: pd.DataFrame, te_values: dict) -> pd.DataFrame:
    for c in TE_KEYS:
        X[f"te_{c}"] = te_values[c].to_numpy()
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _base_features(df)
    keys = _te_key(df)
    te_values = {c: keys[c].map(te_maps_full[c]) for c in TE_KEYS}
    for c in TE_KEYS:
        te_values[c] = te_values[c].fillna(train_prior)
    X = _add_cat(X, df)
    return _add_te(X, te_values)


def prepare_train_oof(df: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    """Same as prepare(), but TE columns are out-of-fold (no leakage into training rows)."""
    X = _base_features(df)
    keys = _te_key(df)
    oof = {c: pd.Series(np.nan, index=df.index) for c in TE_KEYS}
    for tr_idx, va_idx in KFold(n_splits, shuffle=True, random_state=SEED).split(df):
        for c in TE_KEYS:
            m = _te_map(keys[c].iloc[tr_idx], y[tr_idx], TE_M[c])
            oof[c].iloc[va_idx] = keys[c].iloc[va_idx].map(m).to_numpy()
    for c in TE_KEYS:
        oof[c] = oof[c].fillna(train_prior)
    X = _add_cat(X, df)
    return _add_te(X, oof)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
y_train = to_y(train)
train_prior = float(y_train.mean())
te_maps_full = {c: _te_map(k, y_train, TE_M[c]) for c, k in _te_key(train).items()}

N_MODELS = 8
base_params = dict(
    n_estimators=200,
    learning_rate=0.05,
    reg_lambda=1.0,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
)
GRID = [  # (max_depth, colsample_bytree, subsample)
    (6, 0.8, 0.8), (5, 0.9, 0.9), (7, 0.7, 0.8), (6, 0.9, 0.7),
    (5, 0.7, 0.9), (7, 0.8, 0.9), (6, 0.7, 0.7), (8, 0.9, 0.8),
]
VARIATIONS = [(d, cs, ss, use_cats) for use_cats in (True, False) for (d, cs, ss) in GRID]
DROP_CAT_COLS = [c for c in CAT_COLS]

t0 = time.time()
Xtr = prepare_train_oof(train, y_train)
MODELS = []
for k, (d, cs, ss, use_cats) in enumerate(VARIATIONS):
    cols = None if use_cats else [c for c in Xtr.columns if c not in DROP_CAT_COLS]
    m = xgb.XGBClassifier(max_depth=d, colsample_bytree=cs, subsample=ss,
                          random_state=SEED + k, n_jobs=N_JOBS, **base_params)
    m.fit(Xtr if cols is None else Xtr[cols], y_train)
    MODELS.append((m, cols))
print(f"Training time: {time.time() - t0:.1f}s ({len(MODELS)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xq = prepare(df)
    p = np.zeros(len(Xq))
    for m, cols in MODELS:
        p += m.predict_proba(Xq if cols is None else Xq[cols])[:, 1]
    return p / len(MODELS)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
