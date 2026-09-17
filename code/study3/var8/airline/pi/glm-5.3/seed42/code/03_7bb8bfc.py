"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

This experiment: diagnostic ablation of feature sets x tree counts at baseline params.
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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_eval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

BASE_OBJ = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
base_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_OBJ}
route_levels = pd.Index(
    sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique())
)


def feats_A(df):
    """Baseline features."""
    X = pd.DataFrame(index=df.index)
    for c in BASE_OBJ:
        X[c] = pd.Categorical(df[c], categories=base_levels[c])
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["Distance"] = df["Distance"].astype(float)
    return X


def _tod(df):
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 24)
    minute = (dt % 100).clip(0, 59)
    return dt, hour, minute


def feats_B(df):
    """A + departure-time decomposition + cyclical."""
    X = feats_A(df)
    dt, hour, minute = _tod(df)
    X["hour"] = hour
    X["minute"] = minute
    tod = hour * 60 + minute
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    return X


def feats_C(df):
    """B + route + log distance + month cycle."""
    X = feats_B(df)
    X["route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels
    )
    X["log_distance"] = np.log1p(X["Distance"])
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    return X


PARAMS = dict(
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
results = {}
for name, prep in [("A", feats_A), ("B", feats_B), ("C", feats_C)]:
    Xtr, Xev = prep(train), prep(evald)
    for n in (30, 100, 300):
        m = xgb.XGBClassifier(n_estimators=n, **PARAMS)
        m.fit(Xtr, y_train, verbose=False)
        auc = roc_auc_score(y_eval, m.predict_proba(Xev)[:, 1])
        results[(name, n)] = (auc, m, prep)
        print(f"DIAG {name} n={n} eval_auc={auc:.4f}  ({time.time() - t0:.0f}s)")

best_key = max(results, key=lambda k: results[k][0])
best_auc, best_model, best_prep = results[best_key]
print(f"BEST {best_key[0]} n={best_key[1]} eval_auc={best_auc:.4f}")
print(f"Diagnostic time: {time.time() - t0:.1f}s")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return best_prep(df)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return best_model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
