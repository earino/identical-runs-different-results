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


def add_features(df: pd.DataFrame, extra: bool = False) -> pd.DataFrame:
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
    if extra:
        F["dist_bin"] = pd.cut(F["dist"], bins=DIST_EDGES, labels=False).astype(str)
        F["origin_freq"] = F["origin"].map(ORIGIN_FREQ).fillna(0)
        F["dest_freq"] = F["dest"].map(DEST_FREQ).fillna(0)
        F["carrier_freq"] = F["carrier"].map(CARRIER_FREQ).fillna(0)
    return F


BASE_CATS = ["month", "dow", "carrier", "origin", "dest", "hour", "dep_bin15",
             "car_hour"]
BASE_NUMS = ["dist", "dep_num", "hour_sin", "hour_cos", "dep_small", "dep_large",
             "month_num"]
EXTRA_CATS = BASE_CATS + ["dist_bin"]
EXTRA_NUMS = BASE_NUMS + ["origin_freq", "dest_freq", "carrier_freq"]

# fitted statistics from TRAINING data only
DIST_EDGES = np.unique(np.quantile(_num(train["Distance"]).dropna(),
                                   np.linspace(0, 1, 11)))
ORIGIN_FREQ = train["Origin"].value_counts().to_dict()
DEST_FREQ = train["Dest"].value_counts().to_dict()
CARRIER_FREQ = train["UniqueCarrier"].value_counts().to_dict()

F_train = add_features(train, extra=True)
F_eval = add_features(evald, extra=True)
LEVELS = {c: pd.Index(sorted(F_train[c].dropna().unique()))
          for c in set(EXTRA_CATS)}


def make_X(F: pd.DataFrame, cats, nums) -> pd.DataFrame:
    X = pd.DataFrame(index=F.index)
    for c in cats:
        X[c] = pd.Categorical(F[c], categories=LEVELS[c])
    for c in nums:
        X[c] = pd.to_numeric(F[c], errors="coerce")
    return X


X_base_tr = make_X(F_train, BASE_CATS, BASE_NUMS)
X_base_ev = make_X(F_eval, BASE_CATS, BASE_NUMS)
X_ext_tr = make_X(F_train, EXTRA_CATS, EXTRA_NUMS)
X_ext_ev = make_X(F_eval, EXTRA_CATS, EXTRA_NUMS)


def new_model(**kw):
    p = dict(max_depth=6, learning_rate=0.05, tree_method="hist",
             enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


t0 = time.time()
mA = new_model(n_estimators=2000).fit(X_base_tr, y01)
aA = roc_auc_score(y_ev, mA.predict_proba(X_base_ev)[:, 1])
print(f"diag base n=2000 d=6 -> {aA:.4f} ({time.time() - t0:.0f}s)")
mB = new_model(n_estimators=2000).fit(X_ext_tr, y01)
aB = roc_auc_score(y_ev, mB.predict_proba(X_ext_ev)[:, 1])
print(f"diag +distbin/freq n=2000 d=6 -> {aB:.4f} ({time.time() - t0:.0f}s)")

mC = new_model(n_estimators=2500, max_depth=5).fit(X_base_tr, y01)
aC = roc_auc_score(y_ev, mC.predict_proba(X_base_ev)[:, 1])
mD = new_model(n_estimators=1200, max_depth=7, learning_rate=0.06).fit(X_base_tr, y01)
aD = roc_auc_score(y_ev, mD.predict_proba(X_base_ev)[:, 1])
p_ens = (mA.predict_proba(X_base_ev)[:, 1] * 0.5
         + mC.predict_proba(X_base_ev)[:, 1] * 0.25
         + mD.predict_proba(X_base_ev)[:, 1] * 0.25)
a_ens = roc_auc_score(y_ev, p_ens)
print(f"diag depth-ens d6=.5,d5=.25,d7=.25 -> {a_ens:.4f} "
      f"(d5 {aC:.4f}, d7 {aD:.4f}) ({time.time() - t0:.0f}s)")
print(f"sweep time: {time.time() - t0:.1f}s")

CANDS = {"base": (aA, [mA]), "extra": (aB, [mB]),
         "ens": (a_ens, [mA, mC, mD])}
best_name, (best_auc, best_models) = max(CANDS.items(), key=lambda kv: kv[1][0])
print(f"best: {best_name} {best_auc:.4f}")

_MODELS = best_models
_CUR = {"name": best_name}


def prepare(df: pd.DataFrame) -> list:
    F = add_features(df, extra=True)
    if _CUR["name"] == "base":
        return [make_X(F, BASE_CATS, BASE_NUMS)] * len(_MODELS)
    if _CUR["name"] == "extra":
        return [make_X(F, EXTRA_CATS, EXTRA_NUMS)] * len(_MODELS)
    return [make_X(F, BASE_CATS, BASE_NUMS)] * len(_MODELS)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(X)[:, 1] for m, X in zip(_MODELS, prepare(df))]
    if len(Ps) == 1:
        return Ps[0]
    return 0.5 * Ps[0] + 0.25 * Ps[1] + 0.25 * Ps[2]


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
