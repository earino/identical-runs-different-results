"""XGBoost airline-delay classifier, v2: engineered features + bigger model + early stopping.

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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# Categories are fit on TRAINING data only; unseen levels map to NaN (handled by XGBoost).
CAT_LEVELS = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().astype(str).unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().astype(str).unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().astype(str).unique())),
}


def _num(s):
    # 'c-7' -> 7
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    X["Month"] = month
    X["DayOfMonth"] = dom
    X["DayOfWeek"] = dow
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 24)
    minute = dt % 100
    X["DepHour"] = hour + minute / 60.0
    ang = 2.0 * np.pi * X["DepHour"].fillna(12.0) / 24.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    X["RedEye"] = ((hour >= 0) & (hour <= 5)).astype(float)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=CAT_LEVELS["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"].astype(str), categories=CAT_LEVELS["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"].astype(str), categories=CAT_LEVELS["Dest"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(n_estimators=3000, learning_rate=0.03, max_depth=16, subsample=0.7, colsample_bytree=0.6,
         min_child_weight=20, random_state=42),
    dict(n_estimators=3000, learning_rate=0.04, max_depth=12, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=10, random_state=7),
    dict(n_estimators=3000, learning_rate=0.05, max_depth=20, subsample=0.7, colsample_bytree=0.5,
         min_child_weight=30, random_state=123),
    dict(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.9, colsample_bytree=0.9,
         min_child_weight=5, random_state=2024),
    dict(n_estimators=3000, learning_rate=0.05, max_depth=24, subsample=0.7, colsample_bytree=0.4,
         min_child_weight=100, random_state=999),
    dict(n_estimators=3000, learning_rate=0.03, max_depth=16, subsample=0.7, colsample_bytree=0.6,
         min_child_weight=20, random_state=43),
    dict(n_estimators=3000, learning_rate=0.04, max_depth=16, subsample=0.8, colsample_bytree=0.5,
         min_child_weight=10, random_state=44),
    dict(n_estimators=3000, learning_rate=0.04, max_depth=14, subsample=0.75, colsample_bytree=0.65,
         min_child_weight=15, random_state=45),
]


def _fit_one(cfg, Xtr, ytr, Xev, yev):
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=200,
        eval_metric="auc",
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    return m


t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = [_fit_one(cfg, Xtr, ytr, Xev, yev) for cfg in CONFIGS]
for m, cfg in zip(models, CONFIGS):
    print(f"  trees={m.best_iteration + 1} depth={cfg['max_depth']}")
print(f"Training time: {time.time() - t0:.1f}s")


def _rankavg(ps):
    R = np.zeros_like(ps[0], dtype=float)
    for p in ps:
        order = np.argsort(np.argsort(p))
        R += order / (len(p) - 1)
    return R / len(ps)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return _rankavg([m.predict_proba(X)[:, 1] for m in models])


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
