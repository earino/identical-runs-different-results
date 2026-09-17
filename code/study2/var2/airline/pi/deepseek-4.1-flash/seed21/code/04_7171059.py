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

ALL_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def _hour_of(df):
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23)


def _route(df):
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


_cnt_origin = train["Origin"].value_counts()
_cnt_dest = train["Dest"].value_counts()
_cnt_carrier = train["UniqueCarrier"].value_counts()
_cnt_route = _route(train).value_counts()
_cnt_oh = pd.concat([train["Origin"], _hour_of(train).rename("hour")], axis=1).value_counts()


def prepare(df, cats):
    X = pd.DataFrame(index=df.index)
    for c in cats:
        X[c] = pd.Categorical(df[c].astype(str), categories=sorted(train[c].astype(str).unique()))
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    tod = hour + (dep % 100).clip(0, 59) / 60.0
    X["DepTime"] = dep
    X["hour"] = hour
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["origin_cnt"] = df["Origin"].map(_cnt_origin).fillna(0).to_numpy()
    X["dest_cnt"] = df["Dest"].map(_cnt_dest).fillna(0).to_numpy()
    X["carrier_cnt"] = df["UniqueCarrier"].map(_cnt_carrier).fillna(0).to_numpy()
    X["route_cnt"] = _route(df).map(_cnt_route).fillna(0).to_numpy()
    oh = pd.concat([df["Origin"], _hour_of(df).rename("hour")], axis=1)
    X["origin_hour_cnt"] = [float(_cnt_oh.get(t, 0.0)) for t in map(tuple, oh.to_numpy())]
    return X


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)
variants = {
    "all": ALL_CAT,
    "no_od": ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier"],
    "no_cal": ["UniqueCarrier", "Origin", "Dest"],
    "carrier_only": ["UniqueCarrier"],
    "none": [],
    "od_only": ["Origin", "Dest"],
}
best = (-1, None)
for nm, cats in variants.items():
    Xtr, Xev = prepare(train, cats), prepare(evald, cats)
    m = xgb.XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          random_state=SEED, n_jobs=N_JOBS)
    m.fit(Xtr, ytr)
    a = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    print(f"{nm}: {a:.4f}")
    if a > best[0]:
        best = (a, nm, cats)
print(f"BEST {best[1]} auc={best[0]:.4f}")

CATS = best[2]
Xtr, Xev = prepare(train, CATS), prepare(evald, CATS)
feature_cols = Xtr.columns.tolist()
model = xgb.XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          random_state=SEED, n_jobs=N_JOBS)
model.fit(Xtr, ytr)


def predict_proba(df):
    return model.predict_proba(prepare(df, CATS))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
