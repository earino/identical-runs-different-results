"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

BEST_K = 100
N_SEEDS = 8


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _num(df["Month"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    dep = X["DepTime"]
    hour = np.clip((dep // 100) % 24, 0, 23)
    minute = dep % 100
    X["DepHour"] = hour
    X["DepMinute"] = minute
    X["DepTimeMin"] = hour * 60 + minute
    X["sin_day"] = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
    X["cos_day"] = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=BEST_K,
    max_depth=5,
    min_child_weight=30,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr, X_ev = prepare(train), prepare(evald)
models = []
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(**{**PARAMS, "random_state": SEED + s})
    m.fit(X_tr, y_tr, verbose=False)
    models.append(m)


def _avg_proba(X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


for n in [1, 2, 4, N_SEEDS]:
    p = np.mean([m.predict_proba(X_ev)[:, 1] for m in models[:n]], axis=0)
    print(f"[diag] ensemble of {n}: eval_auc={roc_auc_score(y_ev, p):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return _avg_proba(prepare(df))


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
