"""XGBoost binary classifier for airline departure-delay prediction (dep_delayed_15min).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare(), so it transfers to unseen rows.
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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "Distance"]

# fitted on TRAIN ONLY, then reused verbatim by prepare() on any unseen dataframe
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df):
    return df["DepTime"].to_numpy() // 100


def _carrier_hour(df):
    return df["UniqueCarrier"].astype(str) + "_" + _hour(df).astype(str)


ch_levels = pd.Index(sorted(_carrier_hour(train).unique()))


def _ordinal(df, col):
    """c-<n> string level -> integer n."""
    return df[col].astype(str).str.slice(2).astype(int).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["carrier_hour"] = pd.Categorical(_carrier_hour(df), categories=ch_levels)

    d = df["DepTime"].to_numpy()
    h, m = d // 100, d % 100
    tod = h * 60 + m
    dow = _ordinal(df, "DayOfWeek")
    mon = _ordinal(df, "Month")

    X["DepTime"] = d
    X["Distance"] = df["Distance"].to_numpy()
    X["dep_hour"] = h
    X["dep_min"] = m
    X["dep_tod"] = tod
    X["sin_h"] = np.sin(2 * np.pi * h / 24.0)
    X["cos_h"] = np.cos(2 * np.pi * h / 24.0)
    X["sin_t"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_t"] = np.cos(2 * np.pi * tod / 1440.0)
    X["dow"] = dow
    X["mon"] = mon
    X["dom"] = _ordinal(df, "DayofMonth")
    X["is_weekend"] = (dow >= 6).astype(int)
    X["logdist"] = np.log1p(X["Distance"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Ensemble of XGBoost models with deliberately different tree structures; averaging them is
# more robust on unseen (later-year) rows than any single configuration.
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
    dict(max_depth=8, colsample_bylevel=0.7, reg_alpha=1.0),      # second seed
    dict(max_depth=10, colsample_bylevel=0.6, min_child_weight=5),
    dict(max_depth=9, colsample_bylevel=0.65, min_child_weight=3),   # second seed of member 2
]
SEEDS = [42, 43, 44, 142, 45, 243]


def build_models():
    models = []
    for i, p in enumerate(PARAMS):
        kw = dict(BASE)
        kw.update(p)
        kw["random_state"] = SEEDS[i]
        models.append(xgb.XGBClassifier(**kw))
    return models


Xt = prepare(train)
yt = to_y(train)

models = build_models()
t0 = time.time()
for m in models:
    m.fit(Xt, yt)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
