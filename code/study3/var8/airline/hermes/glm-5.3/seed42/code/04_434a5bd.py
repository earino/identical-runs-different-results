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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
RAW_NUM = ["DepTime", "DepMinutes", "Distance"]
RAW_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    dep = df["DepTime"].astype(int)
    X["DepMinutes"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = df["Distance"].astype(float)
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

# --- 3-fold CV on TRAIN ONLY (2005) to pick member hyperparams ----------------
GRID = [
    dict(max_depth=6, min_child_weight=1),
    dict(max_depth=8, min_child_weight=1),
    dict(max_depth=10, min_child_weight=1),
    dict(max_depth=8, min_child_weight=5),
    dict(max_depth=6, min_child_weight=5, gamma=0.5),
    dict(max_depth=10, min_child_weight=3, subsample=0.9),
]
from sklearn.model_selection import KFold

kf = KFold(n_splits=3, shuffle=True, random_state=0)
folds = list(kf.split(Xtr))
cv_scores = {}
t0 = time.time()
for gi, g in enumerate(GRID):
    aucs = []
    for tr_idx, va_idx in folds:
        m = xgb.XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            subsample=g.get("subsample", 0.7),
            colsample_bytree=0.7,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="auc",
            early_stopping_rounds=40,
            random_state=0,
            n_jobs=N_JOBS,
            **{k: v for k, v in g.items() if k not in ("subsample",)},
        )
        m.fit(
            Xtr.iloc[tr_idx], ytr[tr_idx],
            eval_set=[(Xtr.iloc[va_idx], ytr[va_idx])],
            verbose=False,
        )
        aucs.append(roc_auc_score(ytr[va_idx], m.predict_proba(Xtr.iloc[va_idx])[:, 1]))
    cv_scores[gi] = float(np.mean(aucs))
    print(f"CV {gi} {g} -> {cv_scores[gi]:.4f}")
best_g = max(cv_scores, key=lambda k: cv_scores[k])
print(f"CV time: {time.time() - t0:.1f}s, best grid {best_g}: {GRID[best_g]}")

# --- bagged ensemble with the CV-picked member params --------------------------
models = []
t0 = time.time()
for seed in range(7):
    m = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=60,
        random_state=seed,
        n_jobs=N_JOBS,
        **{k: v for k, v in GRID[best_g].items() if k != "subsample"},
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")
print(f"best_iterations: {[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
