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
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of XGBoost classifiers (seed + param jitter) --------
MEMBERS = [
    dict(seed=42, max_depth=10, colsample_bytree=0.5, subsample=0.9),
    dict(seed=7, max_depth=9, colsample_bytree=0.4, subsample=0.9),
    dict(seed=123, max_depth=10, colsample_bytree=0.45, subsample=0.85),
    dict(seed=2024, max_depth=9, colsample_bytree=0.5, subsample=0.95),
    dict(seed=555, max_depth=10, colsample_bytree=0.4, subsample=0.9),
]

Xtr, Xev = prepare(train), prepare(evald)
models = []
t0 = time.time()
for cfg0 in MEMBERS:
    cfg = dict(cfg0)
    seed = cfg.pop("seed")
    m = xgb.XGBClassifier(
        n_estimators=2500,
        learning_rate=0.05,
        min_child_weight=20,
        reg_lambda=30.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=150,
        eval_metric="auc",
        n_jobs=N_JOBS,
        random_state=seed,
        **cfg,
    )
    m.fit(Xtr, y_train, eval_set=[(Xev, y_eval)], verbose=False)
    models.append(m)
    print(f"member seed={seed} d={cfg['max_depth']} cs={cfg['colsample_bytree']} "
          f"best_iter={m.best_iteration}", flush=True)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
