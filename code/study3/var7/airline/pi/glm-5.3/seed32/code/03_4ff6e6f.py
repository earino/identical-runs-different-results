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
y_all = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_all.mean())


def _num(col: pd.Series) -> pd.Series:
    """c-7 -> 7 (int), numeric passthrough otherwise."""
    if col.dtype == object or str(col.dtype) == "string":
        return pd.to_numeric(col.astype(str).str.replace("c-", "", regex=False), errors="coerce")
    return col


def _te_map(keys: pd.Series, m: float) -> dict:
    """Smoothed target-encoding map fitted on the training labels only."""
    g = y_all.groupby(keys.values)
    s, c = g.sum(), g.size()
    te = (s + PRIOR * m) / (c + m)
    return te.to_dict()


month = _num(train["Month"])
dom = _num(train["DayofMonth"])
dow = _num(train["DayOfWeek"])
hour = train["DepTime"] // 100

TE_M = {"UniqueCarrier": 20.0, "Origin": 20.0, "Dest": 20.0, "hour": 50.0}
te_maps = {c: _te_map(train[c], m) for c, m in TE_M.items()}
te_maps["route"] = _te_map(train["Origin"].astype(str) + "-" + train["Dest"].astype(str), 20.0)

# native categoricals fitted on train only
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "-" + train["Dest"].astype(str)).unique()))
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    h = dt // 100
    mi = dt % 100
    mo = _num(df["Month"])
    do = _num(df["DayofMonth"])
    dw = _num(df["DayOfWeek"])
    dist = pd.to_numeric(df["Distance"], errors="coerce")

    X["DepTime"] = dt
    X["hour"] = h
    X["minute"] = mi
    X["hour_sin"] = np.sin(2 * np.pi * h / 24)
    X["hour_cos"] = np.cos(2 * np.pi * h / 24)
    X["month_sin"] = np.sin(2 * np.pi * mo / 12)
    X["month_cos"] = np.cos(2 * np.pi * mo / 12)
    X["dom_sin"] = np.sin(2 * np.pi * do / 31)
    X["dom_cos"] = np.cos(2 * np.pi * do / 31)
    X["dow_sin"] = np.sin(2 * np.pi * dw / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dw / 7)
    X["Distance"] = dist
    X["log_distance"] = np.log1p(dist)

    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    X["route_cat"] = pd.Categorical(route, categories=route_levels)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN

    # target encodings (maps fitted on train only; unseen keys -> PRIOR)
    X["te_carrier"] = df["UniqueCarrier"].map(te_maps["UniqueCarrier"]).fillna(PRIOR)
    X["te_origin"] = df["Origin"].map(te_maps["Origin"]).fillna(PRIOR)
    X["te_dest"] = df["Dest"].map(te_maps["Dest"]).fillna(PRIOR)
    X["te_route"] = route.map(te_maps["route"]).fillna(PRIOR)
    X["te_hour"] = h.map(te_maps["hour"]).fillna(PRIOR)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
