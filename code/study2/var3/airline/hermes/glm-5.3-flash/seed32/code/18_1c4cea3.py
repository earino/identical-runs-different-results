"""XGBoost ensemble: hyperparam- and feature-diverse members, equal-weight average.

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


# fitted statistic from TRAINING data only
DIST_EDGES = np.unique(np.quantile(_num(train["Distance"]).dropna(),
                                   np.linspace(0, 1, 11)))


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
    F["car_hour"] = F["carrier"] + "|" + hh.fillna(-1).astype(int).astype(str)
    F["dep_bin15"] = (minutes // 15).astype(int).astype(str)
    F["hour_sin"] = np.sin(ang)
    F["hour_cos"] = np.cos(ang)
    F["dep_num"] = dep
    F["dep_small"] = (dep < 100).astype(float)
    F["dep_large"] = (dep >= 2400).astype(float)
    F["month_num"] = _num(F["month"].str.slice(2))
    F["dist"] = _num(df["Distance"])
    if extra:
        F["orig_hour"] = F["origin"] + "|" + hh.fillna(-1).astype(int).astype(str)
        F["dist_bin"] = pd.cut(F["dist"], bins=DIST_EDGES, labels=False)
        F["dist_hour"] = F["dist_bin"].astype(str) + "|" + hh.fillna(-1).astype(int).astype(str)
    return F


BASE_CATS = ["month", "dow", "carrier", "origin", "dest", "hour", "dep_bin15",
             "car_hour"]
BASE_NUMS = ["dist", "dep_num", "hour_sin", "hour_cos", "dep_small", "dep_large",
             "month_num"]

F_train = add_features(train, extra=True)
F_eval = add_features(evald, extra=True)
ALL_CATS = set(BASE_CATS + ["orig_hour", "dist_hour"])
LEVELS = {c: pd.Index(sorted(F_train[c].dropna().unique())) for c in ALL_CATS}


def make_X(F: pd.DataFrame, cats) -> pd.DataFrame:
    X = pd.DataFrame(index=F.index)
    for c in cats:
        X[c] = pd.Categorical(F[c], categories=LEVELS[c])
    for c in BASE_NUMS:
        X[c] = pd.to_numeric(F[c], errors="coerce")
    return X


Xb_train = make_X(F_train, BASE_CATS)
Xb_eval = make_X(F_eval, BASE_CATS)


def new_model(**kw):
    p = dict(learning_rate=0.05, tree_method="hist", enable_categorical=True,
             random_state=SEED, n_jobs=N_JOBS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


t0 = time.time()
MEMBERS = []  # (model, cats)

m = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0).fit(Xb_train, y01)
MEMBERS.append((m, BASE_CATS, "d6a"))
print(f"  member d6a      -> {roc_auc_score(y_ev, m.predict_proba(Xb_eval)[:, 1]):.4f} ({time.time() - t0:.0f}s)")

m = new_model(n_estimators=1500, max_depth=4).fit(Xb_train, y01)
MEMBERS.append((m, BASE_CATS, "d4"))
print(f"  member d4       -> {roc_auc_score(y_ev, m.predict_proba(Xb_eval)[:, 1]):.4f} ({time.time() - t0:.0f}s)")

m = new_model(n_estimators=1000, max_depth=8, learning_rate=0.06).fit(Xb_train, y01)
MEMBERS.append((m, BASE_CATS, "d8"))
print(f"  member d8       -> {roc_auc_score(y_ev, m.predict_proba(Xb_eval)[:, 1]):.4f} ({time.time() - t0:.0f}s)")

m = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0, colsample_bynode=0.7).fit(Xb_train, y01)
MEMBERS.append((m, BASE_CATS, "colnode"))
print(f"  member colnode  -> {roc_auc_score(y_ev, m.predict_proba(Xb_eval)[:, 1]):.4f} ({time.time() - t0:.0f}s)")

w = 1.0 + 0.5 * (F_train["month_num"].fillna(1) - 1) / 11.0
m = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0).fit(Xb_train, y01, sample_weight=w)
MEMBERS.append((m, BASE_CATS, "recency"))
print(f"  member recency  -> {roc_auc_score(y_ev, m.predict_proba(Xb_eval)[:, 1]):.4f} ({time.time() - t0:.0f}s)")

for ex in ("orig_hour", "dist_hour"):
    cats = BASE_CATS + [ex]
    m = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0).fit(
        make_X(F_train, cats), y01)
    p = m.predict_proba(make_X(F_eval, cats))[:, 1]
    MEMBERS.append((m, cats, "x_" + ex))
    print(f"  member x_{ex:9s} -> {roc_auc_score(y_ev, p):.4f} ({time.time() - t0:.0f}s)")

print(f"sweep time: {time.time() - t0:.1f}s")

# fixed equal-weight average of all members (chosen by construction, not searched)
_MODELS = [(m, cats) for m, cats, _ in MEMBERS]


def prepare(df: pd.DataFrame) -> list:
    F = add_features(df, extra=True)
    return [make_X(F, cats) for _, cats in _MODELS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(X)[:, 1] for (m, _), X in zip(_MODELS, prepare(df))]
    return np.mean(Ps, axis=0)


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
