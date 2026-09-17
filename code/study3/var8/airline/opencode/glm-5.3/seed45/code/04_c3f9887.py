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

# target-encoding keys: raw column name -> function extracting the key from a row
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "hour", "dow"]
TE_M = 50  # smoothing prior count


def _num(s: pd.Series) -> pd.Series:
    """c-7 -> 7 (Month/DayofMonth/DayOfWeek arrive as c-<n> strings)."""
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def _te_key(df: pd.DataFrame) -> dict:
    """Extract the (categorical) key series TE is computed on, from raw columns."""
    dep = df["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 24)
    return {
        "UniqueCarrier": df["UniqueCarrier"].astype(str),
        "Origin": df["Origin"].astype(str),
        "Dest": df["Dest"].astype(str),
        "hour": hour,
        "dow": df["DayOfWeek"].astype(str),
    }


def _te_map(keys: pd.Series, y: np.ndarray) -> pd.Series:
    """Smoothed target mean per key, fit on given rows only."""
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + TE_M * y.mean()) / (g["count"] + TE_M)


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
            m = _te_map(keys[c].iloc[tr_idx], y[tr_idx])
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
te_maps_full = {c: _te_map(k, y_train) for c, k in _te_key(train).items()}

N_MODELS = 5
base_params = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
)

t0 = time.time()
Xtr = prepare_train_oof(train, y_train)
MODELS = []
for k in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=SEED + k, n_jobs=N_JOBS, **base_params)
    m.fit(Xtr, y_train)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_MODELS} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xq = prepare(df)
    p = np.zeros(len(Xq))
    for m in MODELS:
        p += m.predict_proba(Xq)[:, 1]
    return p / len(MODELS)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
