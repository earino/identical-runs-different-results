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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
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
    ks = _keys(df)
    for c, vc in FREQ_MAPS.items():
        X["cnt_" + c] = np.log1p(ks[c].map(vc).fillna(0.0).to_numpy(dtype=float))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])   # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
