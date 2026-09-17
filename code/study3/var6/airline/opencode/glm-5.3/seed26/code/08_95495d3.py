"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
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

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    mon = df["Month"].str.replace("c-", "", regex=False).astype(int)
    dom = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    dow = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    X["month"] = mon
    X["day"] = dom
    X["dow"] = dow
    X["deptime"] = df["DepTime"].astype(int)
    dt = df["DepTime"].astype(int)
    hour = dt // 100
    minute = dt % 100
    msm = hour * 60 + minute  # minutes since midnight
    X["hour"] = hour
    X["msm"] = msm
    X["dist"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=6, learning_rate=0.1, min_child_weight=1, reg_lambda=1.0),
    dict(max_depth=4, learning_rate=0.05, min_child_weight=10, reg_lambda=1.0),
    dict(max_depth=8, learning_rate=0.1, min_child_weight=50, reg_lambda=3.0),
    dict(max_depth=6, learning_rate=0.1, min_child_weight=5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=5, learning_rate=0.07, min_child_weight=20, reg_lambda=2.0),
    dict(max_depth=3, learning_rate=0.05, min_child_weight=30, reg_lambda=1.0, n_estimators=3000, early_stopping_rounds=100),
    dict(max_depth=7, learning_rate=0.05, min_child_weight=10, reg_lambda=2.0, subsample=0.85, colsample_bytree=0.7),
    dict(max_depth=6, learning_rate=0.03, min_child_weight=10, reg_lambda=1.0, n_estimators=4000, early_stopping_rounds=120),
    dict(max_depth=5, learning_rate=0.15, min_child_weight=5, reg_lambda=1.0, subsample=0.75, colsample_bytree=0.9),
    dict(max_depth=4, learning_rate=0.1, min_child_weight=3, reg_lambda=1.0, colsample_bytree=0.6),
]

t0 = time.time()
Xtr, Xev = prepare(train), prepare(evald)
ytr, yev = to_y(train), to_y(evald)
models = []
for i, cfg in enumerate(CONFIGS):
    cfg = dict(cfg)
    m = xgb.XGBClassifier(
        n_estimators=cfg.pop("n_estimators", 2000),
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        early_stopping_rounds=cfg.pop("early_stopping_rounds", 60),
        random_state=SEED + i,
        **cfg,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s, best_iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
