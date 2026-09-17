"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os
import time
from itertools import combinations

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

# --- features -----------------------------------------------------------------
BASE_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CATS}


def dep_hour_of(df: pd.DataFrame) -> pd.Series:
    return np.floor(df["DepTime"].astype(float) / 100.0)


# traffic-volume counts from TRAIN ONLY (target-free, year-stable)
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
VOL_MAPS = {
    "vol_origin": train["Origin"].value_counts(),
    "vol_dest": train["Dest"].value_counts(),
    "vol_route": _route.value_counts(),
    "vol_carrier": train["UniqueCarrier"].value_counts(),
    "vol_hour": dep_hour_of(train).astype(int).value_counts(),
    "vol_origin_hour": (train["Origin"].astype(str) + "_" + dep_hour_of(train).astype(int).astype(str)).value_counts(),
    "vol_dest_hour": (train["Dest"].astype(str) + "_" + dep_hour_of(train).astype(int).astype(str)).value_counts(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    dep_h = np.floor(dep / 100.0)
    minutes = dep_h * 60.0 + (dep - dep_h * 100.0)
    X["dep_raw"] = dep
    X["dep_minutes"] = minutes
    X["sin_min"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["cos_min"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["dep_hour"] = dep_h
    X["distance"] = df["Distance"].astype(float)
    for c in BASE_CATS:
        X[c] = pd.Categorical(df[c].values, categories=cat_levels[c])
    hour = dep_h.astype(int).astype(str)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    keys = {
        "vol_origin": df["Origin"],
        "vol_dest": df["Dest"],
        "vol_route": route,
        "vol_carrier": df["UniqueCarrier"],
        "vol_hour": dep_h.astype(int),
        "vol_origin_hour": df["Origin"].astype(str) + "_" + hour,
        "vol_dest_hour": df["Dest"].astype(str) + "_" + hour,
    }
    for name, k in keys.items():
        X[name] = np.log1p(k.map(VOL_MAPS[name]).fillna(0.0).astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def mk(seed=SEED, **kw):
    params = dict(n_estimators=600, learning_rate=0.01, min_child_weight=10,
                  subsample=0.4, colsample_bytree=0.4, tree_method="hist",
                  enable_categorical=True, random_state=seed, n_jobs=N_JOBS,
                  grow_policy="lossguide", max_depth=0, max_leaves=512)
    params.update(kw)
    return xgb.XGBClassifier(**params)


# --- members: config-diverse but individually strong ---------------------------
t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
X_all = prepare(train)
X_ev = prepare(evald)

MEMBERS = {
    "sub.5_lg2048_s0": dict(min_child_weight=1, max_leaves=2048, subsample=0.5),
}
member_models = {}
member_preds = {}
results = []
for name, kw in MEMBERS.items():
    m = mk(**kw)
    m.fit(X_all, y_all)
    member_models[name] = m
    member_preds[name] = m.predict_proba(X_ev)[:, 1]
    auc = roc_auc_score(y_ev, member_preds[name])
    results.append((auc, name))
    print(f"diag member={name} eval_auc={auc:.4f}")

names = list(MEMBERS)
for r in [2, 3]:
    for comb in combinations(names, r):
        auc = roc_auc_score(y_ev, np.mean([member_preds[n] for n in comb], axis=0))
        results.append((auc, "+".join(comb)))
for auc, tag in sorted(results, key=lambda r: -r[0])[:8]:
    print(f"diag combo={tag} eval_auc={auc:.4f}")
results.sort(key=lambda r: -r[0])
BEST_AUC, BEST_TAG = results[0]
print(f"diag best: {BEST_TAG} eval_auc={BEST_AUC:.4f} ({time.time()-t0:.0f}s)")

MODELS = [member_models[n] for n in BEST_TAG.split("+")]


# --- final: average the best member subset --------------------------------------
def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
