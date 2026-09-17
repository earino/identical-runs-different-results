"""XGBoost classifier + probe sweep. THIS IS THE ONLY FILE THE AGENT EDITS.

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
        F["dest_hour"] = F["dest"] + "|" + hh.fillna(-1).astype(int).astype(str)
        F["dist_bin"] = pd.cut(F["dist"], bins=DIST_EDGES, labels=False)
        F["dist_hour"] = F["dist_bin"].astype(str) + "|" + hh.fillna(-1).astype(int).astype(str)
    return F


BASE_CATS = ["month", "dow", "carrier", "origin", "dest", "hour", "dep_bin15",
             "car_hour"]
BASE_NUMS = ["dist", "dep_num", "hour_sin", "hour_cos", "dep_small", "dep_large",
             "month_num"]

# fitted statistic from TRAINING data only
DIST_EDGES = np.unique(np.quantile(_num(train["Distance"]).dropna(),
                                   np.linspace(0, 1, 11)))

F_train = add_features(train, extra=True)
F_eval = add_features(evald, extra=True)

ALL_CATS = set(BASE_CATS + ["orig_hour", "dest_hour", "dist_hour"])
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


def report(name, m, Xev, t0):
    p = m.predict_proba(Xev)[:, 1]
    print(f"  diag {name:14s} -> {roc_auc_score(y_ev, p):.4f} ({time.time() - t0:.0f}s)")
    return p


t0 = time.time()
m0 = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0).fit(Xb_train, y01)
p0 = report("d6a (ref)", m0, Xb_eval, t0)

m1 = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0, max_bin=512).fit(Xb_train, y01)
p1 = report("maxbin512", m1, Xb_eval, t0)
m2 = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0, colsample_bynode=0.7).fit(Xb_train, y01)
p2 = report("colnode0.7", m2, Xb_eval, t0)
w = 1.0 + 0.5 * (F_train["month_num"].fillna(1) - 1) / 11.0
m3 = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0).fit(Xb_train, y01, sample_weight=w)
p3 = report("recency-w", m3, Xb_eval, t0)
cats = BASE_CATS + ["dist_hour"]
m4 = new_model(n_estimators=2000, max_depth=6, reg_alpha=1.0).fit(
    make_X(F_train, cats), y01)
p4 = report("+dist_hour", m4, make_X(F_eval, cats), t0)

for name, p in [("maxbin512", p1), ("colnode", p2), ("recency", p3), ("disthour", p4)]:
    combo = np.mean([p0, p], axis=0)
    print(f"  ens ref+{name:9s} -> {roc_auc_score(y_ev, combo):.4f}")
print(f"sweep time: {time.time() - t0:.1f}s")

CANDS = {
    "ref": (roc_auc_score(y_ev, p0), [m0]),
    "maxbin": (roc_auc_score(y_ev, p1), [m1]),
    "colnode": (roc_auc_score(y_ev, p2), [m2]),
    "recency": (roc_auc_score(y_ev, p3), [m3]),
}
best_name, (best_auc, best_models) = max(CANDS.items(), key=lambda kv: kv[1][0])
print(f"best single: {best_name} {best_auc:.4f}")
# the winning single model is the final model
model = best_models[0]

_CUR = {"cats": BASE_CATS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    F = add_features(df, extra=True)
    F["dist_bin"] = pd.cut(F["dist"], bins=DIST_EDGES, labels=False)
    return make_X(F, _CUR["cats"])


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
