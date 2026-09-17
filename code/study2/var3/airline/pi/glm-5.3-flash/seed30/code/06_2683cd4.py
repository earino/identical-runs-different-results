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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _key(df: pd.DataFrame, cols) -> pd.Series:
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "_" + df[c].astype(str)
    return k


def _hour(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).astype(float)


# cross categoricals fit on train only (drift-free interaction features)
CROSS_SPECS = {
    "carrier_hour": ("UniqueCarrier", "hour"),
    "dow_hour": ("DayOfWeek", "hour"),
}
_train_hour = _hour(train)
cross_levels = {
    name: pd.Index(sorted(_key(train.assign(hour=_train_hour), cols).unique()))
    for name, cols in CROSS_SPECS.items()
}


def _cnum(s: pd.Series) -> pd.Series:
    # "c-4" -> 4.0 ; unseen/garbage -> NaN
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    hour = _hour(df)
    minute = (dep % 100).astype(float)
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    tod = (hour * 60.0 + minute) / 1440.0
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    X["night"] = ((hour <= 4) | (hour >= 22)).astype(float)
    X["Month"] = _cnum(df["Month"])
    X["DayofMonth"] = _cnum(df["DayofMonth"])
    X["DayOfWeek"] = _cnum(df["DayOfWeek"])
    X["weekend"] = df["DayOfWeek"].isin(["c-6", "c-7"]).astype(float)
    dist = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    dfh = df.assign(hour=hour)
    for name, cols in CROSS_SPECS.items():
        X[name] = pd.Categorical(_key(dfh, cols), categories=cross_levels[name])  # unseen -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_EST_MAX = 2000
params = dict(
    n_estimators=N_EST_MAX,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
K = 5
DEPTHS = [6, 6, 8, 8, 10]
fold = np.random.RandomState(SEED).randint(0, K, len(X_all))
models = []
for k in range(K):
    va = fold == k
    m = xgb.XGBClassifier(**{**params, "max_depth": DEPTHS[k], "random_state": SEED + k})
    m.fit(X_all[~va], y_all[~va], eval_set=[(X_all[va], y_all[va])], verbose=False)
    models.append(m)
    print(f"fold {k}: best_iter {m.best_iteration} val_auc {m.best_score:.4f}")
print(f"Training time: {time.time() - t0:.1f}s ({K}-fold bag)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
