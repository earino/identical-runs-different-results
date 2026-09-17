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

CATS = ["UniqueCarrier"]


def _hour_of(df):
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23)


def _route(df):
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


_cnt_origin = train["Origin"].value_counts()
_cnt_dest = train["Dest"].value_counts()
_cnt_carrier = train["UniqueCarrier"].value_counts()
_cnt_route = _route(train).value_counts()
_cnt_oh = pd.concat([train["Origin"], _hour_of(train).rename("hour")], axis=1).value_counts()
# network / structural stats
_net_carrier = train.assign(r=_route(train)).groupby("UniqueCarrier")["r"].nunique()
_net_route = train.assign(r=_route(train)).groupby("r")["UniqueCarrier"].nunique()
_net_origin_dest = train.groupby("Origin")["Dest"].nunique()
_net_dest_origin = train.groupby("Dest")["Origin"].nunique()
_dist_origin = train.assign(d=pd.to_numeric(train["Distance"], errors="coerce")).groupby("Origin")["d"].mean()
_dist_carrier = train.assign(d=pd.to_numeric(train["Distance"], errors="coerce")).groupby("UniqueCarrier")["d"].mean()


def prepare(df, extra=False):
    X = pd.DataFrame(index=df.index)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"].astype(str),
                                        categories=sorted(train["UniqueCarrier"].astype(str).unique()))
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    tod = hour + (dep % 100).clip(0, 59) / 60.0
    X["DepTime"] = dep
    X["hour"] = hour
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["origin_cnt"] = df["Origin"].map(_cnt_origin).fillna(0).to_numpy()
    X["dest_cnt"] = df["Dest"].map(_cnt_dest).fillna(0).to_numpy()
    X["carrier_cnt"] = df["UniqueCarrier"].map(_cnt_carrier).fillna(0).to_numpy()
    X["route_cnt"] = _route(df).map(_cnt_route).fillna(0).to_numpy()
    oh = pd.concat([df["Origin"], _hour_of(df).rename("hour")], axis=1)
    X["origin_hour_cnt"] = [float(_cnt_oh.get(t, 0.0)) for t in map(tuple, oh.to_numpy())]
    if extra:
        r = _route(df)
        X["carrier_network"] = df["UniqueCarrier"].map(_net_carrier).fillna(0).to_numpy()
        X["route_ncarriers"] = r.map(_net_route).fillna(0).to_numpy()
        X["origin_ndest"] = df["Origin"].map(_net_origin_dest).fillna(0).to_numpy()
        X["dest_norigin"] = df["Dest"].map(_net_dest_origin).fillna(0).to_numpy()
        X["origin_mean_dist"] = df["Origin"].map(_dist_origin).fillna(0).to_numpy()
        X["carrier_mean_dist"] = df["UniqueCarrier"].map(_dist_carrier).fillna(0).to_numpy()
    return X


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)
Xtr, Xev = prepare(train, True), prepare(evald, True)

CONFIGS = [
    dict(n_estimators=650, max_depth=5, learning_rate=0.025, subsample=0.8, colsample_bytree=0.6, random_state=13),
    dict(n_estimators=650, max_depth=5, learning_rate=0.025, subsample=0.8, colsample_bytree=0.6, random_state=42),
    dict(n_estimators=650, max_depth=5, learning_rate=0.025, subsample=0.8, colsample_bytree=0.6, random_state=1),
    dict(n_estimators=650, max_depth=5, learning_rate=0.025, subsample=0.8, colsample_bytree=0.6, random_state=7),
    dict(n_estimators=650, max_depth=5, learning_rate=0.025, subsample=0.8, colsample_bytree=0.6, random_state=99),
]
models = []
for cfg in CONFIGS:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    models.append(m)

P = np.stack([m.predict_proba(Xev)[:, 1] for m in models], axis=1)
for i, m in enumerate(models):
    print(f"model{i} auc={roc_auc_score(yev, P[:, i]):.4f}")
print(f"ensemble_mean auc={roc_auc_score(yev, P.mean(axis=1)):.4f}")
print(f"ensemble_rank auc={roc_auc_score(yev, np.mean([pd.Series(P[:, i]).rank().to_numpy() for i in range(P.shape[1])], axis=0)):.4f}")


def predict_proba(df):
    X = prepare(df, True)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
