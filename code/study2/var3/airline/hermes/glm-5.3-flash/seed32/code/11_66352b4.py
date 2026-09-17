"""XGBoost classifier + ensemble sweep. THIS IS THE ONLY FILE THE AGENT EDITS.

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
NUMS = ["dist", "dep_num", "hour_sin", "hour_cos", "dep_small", "dep_large",
        "month_num"]

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


X_train = make_X(F_train)
X_eval = make_X(F_eval)


def new_model(**kw):
    p = dict(learning_rate=0.05, tree_method="hist", enable_categorical=True,
             random_state=SEED, n_jobs=N_JOBS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


t0 = time.time()
SPECS = {
    "d4": dict(n_estimators=1500, max_depth=4),
    "d5": dict(n_estimators=2500, max_depth=5),
    "d6": dict(n_estimators=2000, max_depth=6),
    "d7": dict(n_estimators=1200, max_depth=7, learning_rate=0.06),
    "d8": dict(n_estimators=1000, max_depth=8, learning_rate=0.06),
    "d6c7": dict(n_estimators=2000, max_depth=6, colsample_bytree=0.7),
    "d6c5": dict(n_estimators=2500, max_depth=6, colsample_bytree=0.5),
    "d6lr8": dict(n_estimators=1200, max_depth=6, learning_rate=0.08),
}
PRED = {}
MODELS = {}
for name, kw in SPECS.items():
    m = new_model(**kw).fit(X_train, y01)
    p = m.predict_proba(X_eval)[:, 1]
    PRED[name] = p
    MODELS[name] = m
    print(f"  member {name:5s} -> {roc_auc_score(y_ev, p):.4f} ({time.time() - t0:.0f}s)")

NAMES = list(SPECS)


def auc_of(combo):
    p = np.mean([PRED[n] for n in combo], axis=0)
    return roc_auc_score(y_ev, p)


combo = []
best_hist = []
while len(combo) < len(NAMES):
    cand, cauc = None, -1
    for n in NAMES:
        if n in combo:
            continue
        a = auc_of(combo + [n])
        if a > cauc:
            cand, cauc = n, a
    combo.append(cand)
    best_hist.append((list(combo), cauc))
    print(f"  greedy +{cand} -> {cauc:.4f}")
print(f"sweep time: {time.time() - t0:.1f}s")

best_combo, best_auc = max(best_hist, key=lambda c: c[1])
print(f"best: {best_combo} -> {best_auc:.4f}")

_MEMBERS = [MODELS[n] for n in best_combo]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return make_X(add_features(df))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in _MEMBERS], axis=0)


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
