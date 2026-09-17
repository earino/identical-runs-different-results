"""Airline delay XGBoost with engineered features.

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

CAT_COLS = ["carrier", "origin", "dest", "hour"]
CNT_SRC = {"cnt_carrier": "UniqueCarrier", "cnt_origin": "Origin", "cnt_dest": "Dest"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    m = df["Month"].str.slice(2).astype(int)
    dom = df["DayofMonth"].str.slice(2).astype(int)
    dow = df["DayOfWeek"].str.slice(2).astype(int)
    dt = df["DepTime"]
    h = dt // 100
    X = pd.DataFrame(index=df.index)
    X["deptime"] = dt
    X["mins"] = h * 60 + dt % 100
    X["dist"] = df["Distance"]
    X["logdist"] = np.log1p(df["Distance"])
    a = 2.0 * np.pi
    X["h_sin"] = np.sin(a * X["mins"] / 1440.0)
    X["h_cos"] = np.cos(a * X["mins"] / 1440.0)
    X["late_night"] = ((dt >= 2400) | (h <= 3)).astype(np.int8)
    X["month"] = m
    X["dom"] = dom
    X["dow"] = dow
    X["carrier"] = df["UniqueCarrier"]
    X["origin"] = df["Origin"]
    X["dest"] = df["Dest"]
    X["hour"] = h
    if CATS:
        for c in CAT_COLS:
            X[c] = pd.Categorical(X[c], categories=CATS[c])
        for feat, src in CNT_SRC.items():
            X[feat] = df[src].map(CNTS[feat]).astype(float).to_numpy()
        rk = df["Origin"] + "_" + df["Dest"]
        X["cnt_route"] = rk.map(CNT_ROUTE).astype(float).fillna(0.0).to_numpy()
    return X


# category levels / counts fitted on TRAIN only
CATS = {}
CNTS = {}
CNT_ROUTE = None
_tmp = prepare(train)
CATS = {c: pd.Index(sorted(_tmp[c].dropna().unique().tolist())) for c in CAT_COLS}
del _tmp
for feat, src in CNT_SRC.items():
    CNTS[feat] = train[src].value_counts()
CNT_ROUTE = train.groupby(train["Origin"] + "_" + train["Dest"], observed=True).size()


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of XGBoost models ----------------------------------
def make_model(**over):
    p = dict(n_estimators=600, learning_rate=0.05, max_depth=20, min_child_weight=3, subsample=0.9,
             colsample_bytree=0.5, max_bin=1024, reg_lambda=2.0, reg_alpha=1.0, tree_method="hist",
             enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    p.update(over)
    return xgb.XGBClassifier(**p)


MODELS = [
    make_model(),
    make_model(max_depth=12, colsample_bytree=0.7, max_bin=512),
    make_model(random_state=7),
]

t0 = time.time()
Xt = prepare(train)
y = to_y(train)
for mod in MODELS:
    mod.fit(Xt, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.mean([mod.predict_proba(X)[:, 1] for mod in MODELS], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
