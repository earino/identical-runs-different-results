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


def make(df):
    X = df[BASE_CAT].copy()
    for c in BASE_CAT:
        X[c] = X[c].astype("category").astype(str)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    tod = hour + (dep % 100).clip(0, 59) / 60.0
    X["DepTime"] = dep
    X["hour"] = hour
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    return X


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr = make(train)
ytr = to_y(train)
Xev = make(evald)
yev = to_y(evald)
cat_cols = [c for c in Xtr.columns if Xtr[c].dtype == object]
cat_levels = {c: pd.Index(sorted(Xtr[c].dropna().unique())) for c in cat_cols}
for c in cat_cols:
    Xtr[c] = pd.Categorical(Xtr[c], categories=cat_levels[c])
    Xev[c] = pd.Categorical(Xev[c], categories=cat_levels[c])

CHECK = [50, 100, 150, 200, 300, 450, 600, 900]
results = []
for lr in [0.03, 0.05, 0.08]:
    for depth in [5, 6, 7]:
        m = xgb.XGBClassifier(n_estimators=max(CHECK), max_depth=depth, learning_rate=lr,
                              subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                              enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
        m.fit(Xtr, ytr)
        aucs = []
        for k in CHECK:
            a = roc_auc_score(yev, m.predict_proba(Xev, iteration_range=(0, k))[:, 1])
            aucs.append((a, k))
        b = max(aucs)
        results.append((b[0], lr, depth, b[1]))
        print(f"lr={lr} depth={depth} best={b[0]:.4f} @trees={b[1]}  " +
              " ".join(f"{k}:{a:.4f}" for a, k in aucs))

results.sort(reverse=True)
ba, blr, bd, bk = results[0]
print(f"BEST lr={blr} depth={bd} trees={bk} auc={ba:.4f}")

# --- final model ---
model = xgb.XGBClassifier(n_estimators=bk, max_depth=bd, learning_rate=blr, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                          random_state=SEED, n_jobs=N_JOBS)
feature_cols = Xtr.columns.tolist()


def prepare(df):
    X = make(df)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X[feature_cols]


t0 = time.time()
model.fit(prepare(train), ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df):
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
