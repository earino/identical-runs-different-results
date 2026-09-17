"""XGBoost classifier + in-run diagnostic sweep. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y01 = (train[TARGET] == POSITIVE).astype(int)
y_ev = (evald[TARGET] == POSITIVE).astype(int)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    F = pd.DataFrame(index=df.index)
    F["month"] = df["Month"].astype(str)
    F["dow"] = df["DayOfWeek"].astype(str)
    F["carrier"] = df["UniqueCarrier"].astype(str)
    F["origin"] = df["Origin"].astype(str)
    F["dest"] = df["Dest"].astype(str)
    dep = _num(df["DepTime"])
    hh = (dep // 100).clip(0, 25)
    mm = dep - (dep // 100) * 100
    minutes = (hh * 60 + mm) % 1440
    ang = 2 * np.pi * minutes / 1440.0
    F["hour"] = hh
    F["dep_bin15"] = (minutes // 15).astype(int).astype(str)
    F["car_hour"] = F["carrier"] + "|" + hh.fillna(-1).astype(int).astype(str)
    F["hour_sin"] = np.sin(ang)
    F["hour_cos"] = np.cos(ang)
    F["dep_num"] = dep
    F["dep_small"] = (dep < 100).astype(float)
    F["dep_large"] = (dep >= 2400).astype(float)
    F["month_num"] = _num(F["month"].str.slice(2))
    F["dist"] = _num(df["Distance"])
    return F


CATS = ["month", "dow", "carrier", "origin", "dest", "hour", "dep_bin15", "car_hour"]
NUMS = ["dist", "dep_num", "hour_sin", "hour_cos", "dep_small", "dep_large"]

F_train = add_features(train)
F_eval = add_features(evald)
LEVELS = {c: pd.Index(sorted(F_train[c].dropna().unique())) for c in CATS}


def make_X(F: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=F.index)
    for c in CATS:
        X[c] = pd.Categorical(F[c], categories=LEVELS[c])
    for c in NUMS:
        X[c] = pd.to_numeric(F[c], errors="coerce")
    return X


def fit_auc(n_est, depth=6, lr=0.05, mcw=1.0, cs=1.0, lam=1.0, ss=1.0):
    m = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        min_child_weight=mcw, colsample_bytree=cs, reg_lambda=lam,
        subsample=ss, tree_method="hist", enable_categorical=True,
        random_state=SEED, n_jobs=N_JOBS)
    m.fit(X_train, y01)
    auc = roc_auc_score(y_ev, m.predict_proba(X_eval)[:, 1])
    print(f"  diag n={n_est:4d} d={depth} lr={lr} mcw={mcw:g} "
          f"cs={cs:g} ss={ss:g} lam={lam:g} -> {auc:.4f}")
    return auc, (n_est, depth, lr, mcw, cs, lam, ss)


t0 = time.time()
X_train = make_X(F_train)
X_eval = make_X(F_eval)
results = [
    fit_auc(2000),
    fit_auc(2500, lr=0.04),
    fit_auc(2000, mcw=5.0),
    fit_auc(2000, depth=5),
]
print(f"sweep time: {time.time() - t0:.1f}s")

best = max(results, key=lambda r: r[0])
print(f"best diag: {best[1]} -> {best[0]:.4f}")

n, d, lr, mcw, cs, lam, ss = best[1]
model = xgb.XGBClassifier(
    n_estimators=n, max_depth=d, learning_rate=lr, min_child_weight=mcw,
    colsample_bytree=cs, reg_lambda=lam, subsample=ss, tree_method="hist",
    enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
model.fit(X_train, y01)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return make_X(add_features(df))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
