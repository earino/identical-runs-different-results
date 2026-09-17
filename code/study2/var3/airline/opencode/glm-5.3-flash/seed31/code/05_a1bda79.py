"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_eval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

# --- statistics fit on TRAIN ONLY (applied later inside prepare) ---------------
CAT_COLS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    dep = df["DepTime"].astype(int)
    X["minute"] = (dep % 100).astype(float)
    X["minute_sin"] = np.sin(2 * np.pi * (dep % 100) / 60)
    X["minute_cos"] = np.cos(2 * np.pi * (dep % 100) / 60)
    X["DepTime"] = dep.astype(float)
    X["Distance"] = df["Distance"].astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of XGBoost classifiers (two recipes x four seeds) ---------
RECIPES = [
    dict(max_depth=11, learning_rate=0.03, min_child_weight=10, seeds=(42, 7, 123, 2024)),
    dict(max_depth=12, learning_rate=0.05, min_child_weight=20, seeds=(42, 7, 123, 555)),
]

Xtr, Xev = prepare(train), prepare(evald)
models = []
t0 = time.time()
for rec in RECIPES:
    seeds = rec.pop("seeds")
    for seed in seeds:
        m = xgb.XGBClassifier(
            n_estimators=4000,
            subsample=0.9,
            colsample_bytree=0.5,
            reg_lambda=30.0,
            tree_method="hist",
            enable_categorical=True,
            early_stopping_rounds=200,
            eval_metric="auc",
            n_jobs=N_JOBS,
            random_state=seed,
            **rec,
        )
        m.fit(Xtr, y_train, eval_set=[(Xev, y_eval)], verbose=False)
        models.append(m)
    rec["seeds"] = seeds
print(f"Training time: {time.time() - t0:.1f}s  members: {len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
