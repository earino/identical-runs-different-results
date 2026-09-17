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


def add_features(df: pd.DataFrame, month_hour: bool = False) -> pd.DataFrame:
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
    if month_hour:
        F["month_hour"] = F["month"] + "|" + hh.fillna(-1).astype(int).astype(str)
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
CATS_MH = CATS + ["month_hour"]


def make_X(F: pd.DataFrame, cats) -> pd.DataFrame:
    X = pd.DataFrame(index=F.index)
    for c in cats:
        X[c] = pd.Categorical(F[c], categories=LEVELS[c])
    for c in NUMS:
        X[c] = pd.to_numeric(F[c], errors="coerce")
    return X


F_train = add_features(train, month_hour=True)
F_eval = add_features(evald, month_hour=True)
LEVELS = {c: pd.Index(sorted(F_train[c].dropna().unique()))
          for c in set(CATS_MH)}


def new_model(**kw):
    p = dict(n_estimators=2000, max_depth=6, learning_rate=0.05,
             tree_method="hist", enable_categorical=True,
             random_state=SEED, n_jobs=N_JOBS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


t0 = time.time()
Xtr = make_X(F_train, CATS)
Xev = make_X(F_eval, CATS)
half = np.random.RandomState(SEED).rand(len(train)) < 0.5

mA = new_model().fit(Xtr[~half], y01[~half])
mB = new_model().fit(Xtr[half], y01[half])
pA_eval = mA.predict_proba(Xev)[:, 1]
pB_eval = mB.predict_proba(Xev)[:, 1]
print(f"diag css A->eval {roc_auc_score(y_ev, pA_eval):.4f} "
      f"B->eval {roc_auc_score(y_ev, pB_eval):.4f} "
      f"ens {roc_auc_score(y_ev, (pA_eval + pB_eval) / 2):.4f} "
      f"corr {np.corrcoef(pA_eval, pB_eval)[0, 1]:.4f} ({time.time() - t0:.0f}s)")

mh = new_model().fit(make_X(F_train, CATS_MH), y01)
print(f"diag month_hour {roc_auc_score(y_ev, mh.predict_proba(make_X(F_eval, CATS_MH))[:, 1]):.4f}")
print(f"sweep time: {time.time() - t0:.1f}s")

model = new_model().fit(Xtr, y01)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return make_X(add_features(df), CATS)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
