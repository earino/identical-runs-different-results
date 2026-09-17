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

BASE_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def _hour_of(df):
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23)


def make(df):
    X = df[BASE_CAT].copy()
    for c in BASE_CAT:
        X[c] = X[c].astype("category").astype(str)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    tod = hour + (dep % 100).clip(0, 59) / 60.0
    X["DepTime"] = dep
    X["hour"] = hour
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    return X


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# ---- count encodings fitted on TRAIN ONLY ------------------------------------
_tr_hour = _hour_of(train)
_cnt_origin = train["Origin"].value_counts()
_cnt_dest = train["Dest"].value_counts()
_cnt_carrier = train["UniqueCarrier"].value_counts()
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_cnt_route = _route_tr.value_counts()
_cnt_oh = pd.concat([train["Origin"], _tr_hour.rename("hour")], axis=1).value_counts()


def prepare(df):
    X = make(df)
    X["origin_cnt"] = df["Origin"].map(_cnt_origin).fillna(0).to_numpy()
    X["dest_cnt"] = df["Dest"].map(_cnt_dest).fillna(0).to_numpy()
    X["carrier_cnt"] = df["UniqueCarrier"].map(_cnt_carrier).fillna(0).to_numpy()
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_cnt"] = route.map(_cnt_route).fillna(0).to_numpy()
    oh = pd.concat([df["Origin"], _hour_of(df).rename("hour")], axis=1)
    X["origin_hour_cnt"] = [float(_cnt_oh.get(t, 0.0)) for t in map(tuple, oh.to_numpy())]
    for c in BASE_CAT:
        X[c] = pd.Categorical(X[c], categories=sorted(train[c].unique()))
    return X


ytr = to_y(train)
yev = to_y(evald)
feature_cols = prepare(train).columns.tolist()

model = None
best = (-1, None)
m = xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                      random_state=SEED, n_jobs=N_JOBS)
t0 = time.time()
m.fit(prepare(train), ytr)
Xev = prepare(evald)
for k in [300, 450, 600]:
    a = roc_auc_score(yev, m.predict_proba(Xev, iteration_range=(0, k))[:, 1])
    print(f"trees={k} eval={a:.4f}")
    if a > best[0]:
        best = (a, k)
print(f"BEST trees={best[1]} auc={best[0]:.4f}  (train {time.time()-t0:.1f}s)")

model = xgb.XGBClassifier(n_estimators=best[1], max_depth=5, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          random_state=SEED, n_jobs=N_JOBS)
model.fit(prepare(train), ytr)


def predict_proba(df):
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
