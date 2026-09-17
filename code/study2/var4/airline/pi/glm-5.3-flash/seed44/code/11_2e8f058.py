"""XGBoost binary classifier, experiment 5: time-of-day + calendar feature engineering.

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
# c-<n> string columns -> numeric; DepTime hhmm -> hour/minute/cyclic; log distance.
NUM_COLS = ["DepTime", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# levels for interaction categoricals, fit on train only (unseen combos -> NaN)
_t = train
_h = (pd.to_numeric(_t["DepTime"], errors="coerce") // 100).clip(0, 29).astype(int).astype(str)
IX_LEVELS = {
    "ix_carrier_hour": pd.Index(sorted((_t["UniqueCarrier"].astype(str) + "_" + _h).unique())),
    "ix_origin_hour": pd.Index(sorted((_t["Origin"].astype(str) + "_" + _h).unique())),
    "ix_dow_hour": pd.Index(sorted((_t["DayOfWeek"].astype(str) + "_" + _h).unique())),
    "ix_carrier_dow": pd.Index(sorted((_t["UniqueCarrier"].astype(str) + "_" + _t["DayOfWeek"].astype(str)).unique())),
}


def _cal_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _cal_num(df["Month"])
    dom = _cal_num(df["DayofMonth"])
    dow = _cal_num(df["DayOfWeek"])
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 29)
    minute = dt % 100
    mins = (hour * 60 + minute).clip(0, 24 * 60 + 59)

    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["DepTime"] = dt
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_dist"] = np.log1p(X["Distance"])
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["doy"] = month * 31 + dom  # rough seasonality proxy
    X["doy_sin"] = np.sin(2 * np.pi * X["doy"] / 372.0)
    X["doy_cos"] = np.cos(2 * np.pi * X["doy"] / 372.0)
    X["is_weekend"] = (dow >= 6).astype(float)
    # interaction categoricals: direct interaction splits, no target leakage
    hour_s = hour.astype(int).astype(str)
    X["ix_carrier_hour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + hour_s,
                                           categories=IX_LEVELS["ix_carrier_hour"])
    X["ix_origin_hour"] = pd.Categorical(df["Origin"].astype(str) + "_" + hour_s,
                                          categories=IX_LEVELS["ix_origin_hour"])
    X["ix_dow_hour"] = pd.Categorical(df["DayOfWeek"].astype(str) + "_" + hour_s,
                                       categories=IX_LEVELS["ix_dow_hour"])
    X["ix_carrier_dow"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + df["DayOfWeek"].astype(str),
                                          categories=IX_LEVELS["ix_carrier_dow"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Ensemble of diverse XGBoost configs (variance reduction for the year-shifted eval).
CONFIGS = [
    dict(n_estimators=30, max_depth=6, learning_rate=0.1),
    dict(n_estimators=40, max_depth=4, learning_rate=0.15),
    dict(n_estimators=50, max_depth=5, learning_rate=0.12, subsample=0.85),
    dict(n_estimators=40, max_depth=7, learning_rate=0.1, colsample_bynode=0.8, subsample=0.85),
    dict(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8),
    dict(n_estimators=60, max_depth=0, max_leaves=31, grow_policy="lossguide", learning_rate=0.1, subsample=0.85),
    dict(n_estimators=80, max_depth=0, max_leaves=64, grow_policy="lossguide", learning_rate=0.1, colsample_bynode=0.7),
    dict(n_estimators=400, max_depth=6, learning_rate=0.4, booster="dart", rate_drop=0.2, skip_drop=0.5),
    dict(n_estimators=250, max_depth=4, learning_rate=0.3, booster="dart", rate_drop=0.1, skip_drop=0.5),
]


def _make(cfg, seed):
    return xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=seed,
                             n_jobs=N_JOBS, **cfg)


t0 = time.time()
Xp = prepare(train)
y = to_y(train)
models = [_make(cfg, seed) for seed, cfg in enumerate(CONFIGS, start=SEED)]
for m in models:
    m.fit(Xp, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.stack([m.predict_proba(X)[:, 1] for m in models])
    eps = 1e-7
    logits = np.log((ps + eps) / (1 - ps + eps))
    return 1.0 / (1.0 + np.exp(-logits.mean(axis=0)))  # logit-average


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
