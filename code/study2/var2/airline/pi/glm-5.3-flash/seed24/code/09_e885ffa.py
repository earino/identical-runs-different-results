"""XGBoost binary classifier on the airline delay task.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# fitted on training data only
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayofMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    X["DepTime"] = df["DepTime"].astype(int)
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["Distance"] = df["Distance"].astype(float)
    X["logDist"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=24, subsample=0.9, colsample_bytree=0.9, learning_rate=0.05, seed=42),
    dict(max_depth=16, subsample=0.8, colsample_bytree=0.7, learning_rate=0.05, seed=7),
    dict(max_depth=20, subsample=0.7, colsample_bytree=0.8, learning_rate=0.05, seed=123),
    dict(max_depth=24, subsample=0.9, colsample_bytree=0.9, learning_rate=0.03, seed=500),
]


def make_model(n_estimators=600, cfg=None, es=True):
    cfg = cfg or {}
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50 if es else None,
        n_jobs=N_JOBS,
        **cfg,
    )

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(X_all))
val_idx, fit_idx = idx[: len(idx) // 5], idx[len(idx) // 5 :]
Xv, yv = X_all.iloc[val_idx], y_all[val_idx]
models = []
for cfg in CONFIGS:
    es = make_model(cfg=cfg)
    es.fit(X_all.iloc[fit_idx], y_all[fit_idx], eval_set=[(Xv, yv)], verbose=False)
    best_n = int(es.best_iteration) + 1
    print(f"cfg={cfg}: {best_n} rounds (val auc {es.best_score:.4f})")
    m = make_model(n_estimators=best_n, cfg=cfg, es=False)
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
