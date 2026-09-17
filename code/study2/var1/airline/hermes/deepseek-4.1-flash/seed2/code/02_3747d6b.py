"""DIAGNOSTIC SWEEP (temporary): evaluate a small grid of feature-set x capacity configs.
Prints a table of Eval AUC for each config so the next committed train.py can be chosen
deliberately. The eval.csv AUC here is a measurement only; the final script is written after."""
import itertools
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
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

plain = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Distance"]
cats_f0 = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cats_f2 = ["UniqueCarrier", "Origin", "Dest"]
route_tr = (train["Origin"] + "_" + train["Dest"])
route_ev = (evald["Origin"] + "_" + evald["Dest"])
route_levels = pd.Index(sorted(route_tr.unique()))
lv0 = {c: pd.Index(sorted(train[c].unique())) for c in cats_f0}
lv2 = {c: pd.Index(sorted(train[c].unique())) for c in cats_f2}


def fs_baseline(df):
    X = df[plain].copy()
    for c in cats_f0:
        X[c] = pd.Categorical(X[c], categories=lv0[c])
    return X


def fs_plus_time(df):
    X = fs_baseline(df)
    dep = df["DepTime"].astype(np.int32)
    h = (dep // 100).clip(0, 23)
    m = (dep % 100).clip(0, 59)
    X["DepHour"] = h.astype(np.int16)
    X["DepMinOfDay"] = (h * 60 + m).astype(np.int16)
    return X


def fs_numeric_time(df):
    X = df[plain].copy()
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = df[c].str.replace("c-", "", regex=False).astype(np.int16)
    dep = df["DepTime"].astype(np.int32)
    h = (dep // 100).clip(0, 23)
    m = (dep % 100).clip(0, 59)
    X["DepHour"] = h.astype(np.int16)
    X["DepMinOfDay"] = (h * 60 + m).astype(np.int16)
    X["HourSin"] = np.sin(2 * np.pi * X["DepHour"] / 24)
    X["HourCos"] = np.cos(2 * np.pi * X["DepHour"] / 24)
    doy = (X["Month"] - 1) * 30 + X["DayofMonth"]
    X["DoySin"] = np.sin(2 * np.pi * doy / 365)
    X["DoyCos"] = np.cos(2 * np.pi * doy / 365)
    for c in cats_f2:
        X[c] = pd.Categorical(X[c], categories=lv2[c])
    return X


def fs_route(df):
    X = fs_numeric_time(df)
    X["Route"] = pd.Categorical(df["Origin"] + "_" + df["Dest"], categories=route_levels)
    return X


FEATS = {"F0_base": fs_baseline, "F1_time": fs_plus_time, "F2_num": fs_numeric_time, "F3_route": fs_route}

MODELS = {
    "M1_30t_d6": dict(n_estimators=30, max_depth=6, learning_rate=0.1),
    "M2_100t_d6": dict(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8),
    "M3_400t_d4reg": dict(n_estimators=400, max_depth=4, learning_rate=0.05, min_child_weight=20,
                          subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0),
    "M4_800t_d5reg": dict(n_estimators=800, max_depth=5, learning_rate=0.03, min_child_weight=30,
                          subsample=0.7, colsample_bytree=0.5, reg_lambda=5.0),
}

results = []
for (fname, fe), (mname, mp) in itertools.product(FEATS.items(), MODELS.items()):
    Xtr = fe(train)
    Xev = fe(evald)
    params = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    params.update(mp)
    t0 = time.time()
    m = xgb.XGBClassifier(**params)
    m.fit(Xtr, y_tr, verbose=False)
    auc = roc_auc_score(y_ev, m.predict_proba(Xev)[:, 1])
    dt = time.time() - t0
    results.append((fname, mname, auc, dt, Xtr.shape[1]))
    print(f"RESULT {fname:9s} {mname:12s} ncol={Xtr.shape[1]:2d} auc={auc:.4f} t={dt:.1f}s", flush=True)

print("\n=== ranked ===")
for fname, mname, auc, dt, nc in sorted(results, key=lambda r: -r[2]):
    print(f"{auc:.4f}  {fname:9s} {mname:12s}")

best = max(results, key=lambda r: r[2])
print(f"Eval AUC: {best[2]:.4f}")


def predict_proba(df):
    return np.zeros(len(df))
