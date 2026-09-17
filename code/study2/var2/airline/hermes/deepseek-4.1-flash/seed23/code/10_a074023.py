"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- schema -------------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
RAW_CAL = ["Month", "DayofMonth", "DayOfWeek"]  # replaced by numeric versions in prepare()
BASE = [c for c in train.columns if c not in ID_COLS + [TARGET] + CAT_COLS + RAW_CAL]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _dep_tod(dep: np.ndarray) -> np.ndarray:
    """Scheduled departure hhmm -> minutes since midnight (24xx..26xx wrap to after midnight)."""
    hour = np.floor(dep / 100.0)
    tod = hour * 60.0 + (dep - hour * 100.0)
    return np.where(tod >= 24 * 60.0, tod - 24 * 60.0, tod)


def _dep_hour(dep: np.ndarray) -> np.ndarray:
    return np.floor(_dep_tod(dep) / 60.0)


def _pair(a: np.ndarray, b: np.ndarray) -> pd.Index:
    return pd.Index(np.asarray(a).astype(str)) + "_" + pd.Index(np.asarray(b).astype(str))


# Categorical levels are taken from TRAIN ONLY; unseen levels become missing at predict time.
HC_LEVELS = pd.Index(sorted(_pair(train["UniqueCarrier"], _dep_hour(train["DepTime"].to_numpy(dtype=float))).unique()))
OH_LEVELS = pd.Index(sorted(_pair(train["Origin"], _dep_hour(train["DepTime"].to_numpy(dtype=float))).unique()))


def _num(series: pd.Series) -> pd.Series:
    """'c-7' -> 7 (calendar fields are stored as c-<n> strings)."""
    return series.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[BASE + CAT_COLS].copy()

    dep = df["DepTime"].to_numpy(dtype=float)
    tod = _dep_tod(dep)
    X["dep_hour"] = _dep_hour(dep)
    X["dep_minute"] = tod % 60.0
    X["dep_tod"] = tod
    X["dep_sin"] = np.sin(2.0 * np.pi * tod / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * tod / 1440.0)

    X["month_n"] = _num(df["Month"])
    X["dom_n"] = _num(df["DayofMonth"])
    X["dow_n"] = _num(df["DayOfWeek"])
    X["is_weekend"] = (X["dow_n"] >= 6).astype(int)
    X["dist_log"] = np.log1p(df["Distance"])

    # explicit interactions: carrier x time-of-day and origin x time-of-day (the strongest engineered features)
    hr = _dep_hour(dep)
    X["hour_carrier"] = pd.Categorical(_pair(df["UniqueCarrier"], hr), categories=HC_LEVELS)
    X["hour_origin"] = pd.Categorical(_pair(df["Origin"], hr), categories=OH_LEVELS)

    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of XGBoost models (variance reduction on a year-shifted target) ----
PARAMS = dict(min_child_weight=20, learning_rate=0.05, subsample=0.8, reg_lambda=20.0,
              tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
ENSEMBLE = [
    dict(n_estimators=400, max_depth=13, colsample_bytree=0.4, random_state=SEED),
    dict(n_estimators=400, max_depth=16, colsample_bytree=0.4, random_state=SEED + 7),
    dict(n_estimators=400, max_depth=13, colsample_bytree=0.3, random_state=SEED + 101),
    dict(n_estimators=400, max_depth=10, colsample_bytree=0.5, random_state=SEED + 202,
         max_cat_to_onehot=8, drop=["hour_origin"]),
    dict(n_estimators=700, max_depth=13, colsample_bytree=0.4, random_state=SEED + 303, learning_rate=0.03),
    dict(n_estimators=400, max_depth=16, colsample_bytree=0.3, random_state=SEED + 404,
         max_cat_threshold=32, drop=["DepTime", "dep_sin", "dep_cos"]),
    # diverse additions: leaf-wise growth and a gamma-regularized depthwise model
    dict(n_estimators=300, max_depth=0, grow_policy="lossguide", max_leaves=512,
         colsample_bytree=0.4, random_state=SEED + 505),
    dict(n_estimators=400, max_depth=13, colsample_bytree=0.4, random_state=SEED + 606, gamma=0.5,
         reg_alpha=1.0),
]
# each member may see a reduced feature view: diversity without extra runtime
members = [(xgb.XGBClassifier(**{**PARAMS, **{k: v for k, v in cfg.items() if k != "drop"}}),
            cfg.get("drop", [])) for cfg in ENSEMBLE]

t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
for m, drop in members:
    m.fit(X_train.drop(columns=drop), y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp.drop(columns=drop))[:, 1] for m, drop in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
