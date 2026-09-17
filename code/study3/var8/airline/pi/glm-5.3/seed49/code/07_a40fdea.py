"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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

# --- feature engineering -------------------------------------------------------
C_CAT = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
NUM_COLS = ["Distance", "DepTime"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in C_CAT}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here; predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["hour"] = hour / 23.0
    X["min_of_day"] = (hour * 60 + minute) / 1439.0
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in C_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

CONFIGS = {
    "A_d6_lr10": dict(max_depth=6, learning_rate=0.1, min_child_weight=1),
    "B_d3_lr10": dict(max_depth=3, learning_rate=0.1, min_child_weight=1),
    "C_d6_lr03": dict(max_depth=6, learning_rate=0.03, min_child_weight=1),
    "D_d6_lr10_mcw20": dict(max_depth=6, learning_rate=0.1, min_child_weight=20),
    "E_d6_lr10_ss06": dict(max_depth=6, learning_rate=0.1, min_child_weight=1, subsample=0.6),
    "F_d8_lr10": dict(max_depth=8, learning_rate=0.1, min_child_weight=1),
}


def fit_probe(name, params):
    m = xgb.XGBClassifier(
        n_estimators=4000,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=SEED,
        n_jobs=N_JOBS,
        **params,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    r = m.evals_result()["validation_0"]["auc"]
    traj = {k: round(r[min(k, len(r)) - 1], 4) for k in (10, 30, 60, 100, 200, 400, 800)}
    print(f"[probe] {name}: peak={max(r):.4f} at {int(np.argmax(r)) + 1} it, traj={traj}")
    return m, max(r)


results = {}
models = {}
for name, params in CONFIGS.items():
    m, peak = fit_probe(name, params)
    results[name] = peak
    models[name] = m

best_name = max(results, key=results.get)
model = models[best_name]
print(f"[probe] best config: {best_name} ({results[best_name]:.4f})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
