"""XGBoost binary classifier + feature engineering. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Exp 3: ablations of the exp-2 feature set with a fixed-round model (no early stopping on a random split).
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

# ---------------------------------------------------------------- config -----
USE_TIME = True          # scheduled departure time-of-day features
USE_ROUTE = False        # Origin_Dest / carrier_origin high-cardinality categoricals
USE_LOG_DIST = True
USE_DENSITY = True       # airport-hour traffic share features
MAX_CARD = 5000
# -----------------------------------------------------------------------------

CAT_RAW = ["Month", "DayofMonth", "DayOfWeek"]


def _base(df: pd.DataFrame) -> pd.DataFrame:
    """Raw -> engineered frame. Pure per-row transforms only (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    if USE_LOG_DIST:
        X["log_dist"] = np.log1p(dist)

    if USE_TIME:
        dep = pd.to_numeric(df["DepTime"], errors="coerce")
        dep = dep.where((dep >= 0) & (dep <= 2400))
        hour = np.floor(dep / 100.0).clip(0, 23)
        minute = dep - hour * 100
        minute = minute.where((minute >= 0) & (minute < 60))
        tod = hour + minute / 60.0
        X["dep_hour"] = hour
        X["tod"] = tod
        X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
        X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)

    for c in CAT_RAW:
        X[c.lower()] = df[c].astype(str).replace({"nan": np.nan})
    X["carrier"] = df["UniqueCarrier"].astype(str).replace({"nan": np.nan})
    X["origin"] = df["Origin"].astype(str).replace({"nan": np.nan})
    X["dest"] = df["Dest"].astype(str).replace({"nan": np.nan})
    if USE_ROUTE:
        r = X["origin"].fillna("?") + "_" + X["dest"].fillna("?")
        X["route"] = r
        X["carrier_origin"] = X["carrier"].fillna("?") + "_" + X["origin"].fillna("?")
    if USE_DENSITY:
        # how busy is this airport at this departure hour, as a share of that airport's departures
        # (scale-free: works the same whether the frame holds 100k rows or the 1M-row holdout)
        hr = X["dep_hour"].astype("Int64").astype(str)
        oh = X["origin"].fillna("?") + "_" + hr
        X["origin_hour_share"] = (oh.map(oh.value_counts()) / X["origin"].map(X["origin"].value_counts())).to_numpy(dtype=float)
        dh = X["dest"].fillna("?") + "_" + hr
        X["dest_hour_share"] = (dh.map(dh.value_counts()) / X["dest"].map(X["dest"].value_counts())).to_numpy(dtype=float)
    return X


NUMERIC = {"Distance", "log_dist", "dep_hour", "tod", "tod_sin", "tod_cos",
           "origin_hour_share", "dest_hour_share"}

_probe = _base(train)
ALL_COLS = list(_probe.columns)
cat_levels = {}
for c in ALL_COLS:
    if c in NUMERIC:
        continue
    vals = _probe[c].dropna().unique()
    if len(vals) <= MAX_CARD:
        cat_levels[c] = pd.Index(np.sort(vals))
feature_cols = [c for c in ALL_COLS if c in NUMERIC or c in cat_levels]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = _base(df)[feature_cols].copy()
    for c in feature_cols:
        if c in cat_levels:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models (ensemble of XGBoost variants) ------------------------------------
BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=4,
    n_jobs=N_JOBS,
)
PARAM_SETS = [
    dict(max_depth=4, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=10, reg_lambda=1.0, random_state=SEED + 0),
    dict(max_depth=5, learning_rate=0.05, n_estimators=300, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=10, reg_lambda=1.0, random_state=SEED + 1),
    dict(max_depth=6, learning_rate=0.04, n_estimators=400, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=20, reg_lambda=2.0, random_state=SEED + 2),
    dict(max_depth=4, learning_rate=0.03, n_estimators=600, subsample=0.7, colsample_bytree=0.6,
         min_child_weight=20, reg_lambda=2.0, random_state=SEED + 3),
]

Xt = prepare(train)
yt = to_y(train)

models = []
t0 = time.time()
for ps in PARAM_SETS:
    m = xgb.XGBClassifier(**BASE, **ps)
    m.fit(Xt, yt, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  n_models={len(models)}  n_features={len(feature_cols)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dmat = xgb.DMatrix(prepare(df), enable_categorical=True)
    preds = np.mean([m.get_booster().predict(dmat) for m in models], axis=0)
    return preds


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
