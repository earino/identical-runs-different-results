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


def _cint(s):
    return s.str.replace("c-", "", regex=False).astype(float)


def make(df, use_time=False, use_logdist=False, use_route=False):
    X = df[[c for c in BASE_CAT if c in df.columns]].copy()
    for c in BASE_CAT:
        X[c] = X[c].astype("category").astype(str)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    if use_time:
        dep = X["DepTime"]
        hour = (dep // 100).clip(0, 23)
        tod = hour + (dep % 100).clip(0, 59) / 60.0
        X["hour"] = hour
        X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
        X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    if use_logdist:
        X["log_distance"] = np.log1p(X["Distance"])
    if use_route:
        X["route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str))
    return X


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def params(depth=6):
    return dict(n_estimators=150, max_depth=depth, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                random_state=SEED, n_jobs=N_JOBS)


def evaluate(name, use_time, use_logdist, use_route):
    Xtr, Xev = make(train, use_time, use_logdist, use_route), make(evald, use_time, use_logdist, use_route)
    cats = [c for c in Xtr.columns if Xtr[c].dtype == object]
    for c in cats:
        levels = pd.Index(sorted(Xtr[c].dropna().unique()))
        Xtr[c] = pd.Categorical(Xtr[c], categories=levels)
        Xev[c] = pd.Categorical(Xev[c], categories=levels)
    m = xgb.XGBClassifier(**params())
    m.fit(Xtr, to_y(train))
    auc = roc_auc_score(to_y(evald), m.predict_proba(Xev)[:, 1])
    print(f"{name}: {auc:.4f}")
    return auc, Xtr.columns.tolist(), cats


configs = {
    "A_base": (False, False, False),
    "B_time": (True, False, False),
    "C_logdist": (False, True, False),
    "D_time_logdist": (True, True, False),
    "E_time_logdist_route": (True, True, True),
}
best = (-1, None, None, None)
for nm, (ut, ul, ur) in configs.items():
    auc, cols, cats = evaluate(nm, ut, ul, ur)
    if auc > best[0]:
        best = (auc, nm, (ut, ul, ur), cols)
print(f"BEST {best[1]} auc={best[0]:.4f}")

# --- final model with best feature config ---
ut, ul, ur = best[2]
Xtr = make(train, ut, ul, ur)
cat_cols = [c for c in Xtr.columns if Xtr[c].dtype == object]
cat_levels = {c: pd.Index(sorted(Xtr[c].dropna().unique())) for c in cat_cols}
feature_cols = Xtr.columns.tolist()


def prepare(df):
    X = make(df, ut, ul, ur)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X[feature_cols]


model = xgb.XGBClassifier(**params())
t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df):
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
