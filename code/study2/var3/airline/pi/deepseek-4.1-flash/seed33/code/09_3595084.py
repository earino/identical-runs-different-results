"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in `prepare()`; every fitted statistic (categories, frequency maps) is
learned from `data/train.csv` only and stored in STATE, so `predict_proba` reproduces it on unseen rows.
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "hour"]
FREQ_KEYS = [
    "Origin",
    "Dest",
    "UniqueCarrier",
    "route",
    "orig_hour",
    "carrier_origin",
    "dest_hour",
    "carrier_hour",
    "origin_dow",
    "dest_dow",
    "route_hour",
    "carrier_dest",
    "origin_month",
    "dest_month",
]
KEY_ONLY = ["route", "orig_hour", "carrier_origin", "dest_hour", "carrier_hour", "origin_dow", "dest_dow", "route_hour", "carrier_dest", "origin_month", "dest_month"]
STATE = {}


def _derive(df):
    """Row-wise derived columns (no fitting needed)."""
    dep = df["DepTime"].to_numpy(dtype=np.int64)
    hour = (dep // 100) % 24
    minute = dep % 100
    tod = hour + minute / 60.0
    X = pd.DataFrame(index=df.index)
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["minute"] = minute
    X["month"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["dom"] = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["dow"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(X["distance"])
    X["est_arr_tod"] = (tod + X["distance"] / 450.0) % 24.0
    X["hour"] = hour
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["route"] = X["Origin"] + "_" + X["Dest"]
    X["orig_hour"] = X["Origin"] + "_" + hour.astype(str)
    X["carrier_origin"] = X["UniqueCarrier"] + "_" + X["Origin"]
    X["dest_hour"] = X["Dest"] + "_" + hour.astype(str)
    X["carrier_hour"] = X["UniqueCarrier"] + "_" + hour.astype(str)
    X["origin_dow"] = X["Origin"] + "_" + X["dow"].astype(str)
    X["dest_dow"] = X["Dest"] + "_" + X["dow"].astype(str)
    X["route_hour"] = X["route"] + "_" + hour.astype(str)
    X["carrier_dest"] = X["UniqueCarrier"] + "_" + X["Dest"]
    X["origin_month"] = X["Origin"] + "_" + X["month"].astype(str)
    X["dest_month"] = X["Dest"] + "_" + X["month"].astype(str)
    return X


def fit(train_df):
    """Learn categorical levels and frequency maps from the training frame, return its features."""
    X = _derive(train_df)
    STATE["cat_levels"] = {c: pd.Index(sorted(X[c].unique())) for c in CAT_COLS}
    STATE["freq"] = {k: X[k].value_counts() for k in FREQ_KEYS}
    return prepare(train_df)


def prepare(df):
    """Raw DataFrame -> feature frame, using only STATE fitted on training data."""
    X = _derive(df)
    for k in FREQ_KEYS:
        X["freq_" + k] = X[k].map(STATE["freq"][k]).fillna(0).astype(float)
    for c, levels in STATE["cat_levels"].items():
        X[c] = pd.Categorical(X[c], categories=levels)
    return X.drop(columns=KEY_ONLY)


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- fit features, build model ensemble ---------------------------------------
Xtr = fit(train)
ytr = to_y(train)
FEATURES = list(Xtr.columns)
Xev = prepare(evald)[FEATURES]

PARAMS = dict(
    n_estimators=600,
    max_depth=20,
    learning_rate=0.03,
    gamma=1.0,
    reg_lambda=10.0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    n_jobs=N_JOBS,
)
SEEDS = [42, 1, 7]

models = []
t0 = time.time()
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **PARAMS)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df):
    X = prepare(df)[FEATURES]
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
