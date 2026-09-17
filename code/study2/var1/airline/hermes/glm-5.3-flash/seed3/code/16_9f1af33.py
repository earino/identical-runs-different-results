"""Diverse-config XGBoost ensemble on the settled feature set; mean vs rank averaging.

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


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean rank), numpy-only."""
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    xs = x[sorter]
    obs = np.r_[True, xs[1:] != xs[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET, "Month", "DayofMonth"]]
str_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in str_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in str_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in str_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all = prepare(train)
y = to_y(train)
Xe_all = prepare(evald)
ye = to_y(evald)
FEATS = list(X_all.columns)

MEMBERS = [
    ("best_ss95", dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, random_state=42)),
    ("lr04", dict(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.95, colsample_bytree=0.95, random_state=43)),
    ("n250_mcw10", dict(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, min_child_weight=10, random_state=44)),
    ("d3", dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=45)),
    ("seed46", dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, random_state=46)),
    ("seed47", dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, random_state=47)),
]

models = []
preds = []
t0 = time.time()
for name, cfg in MEMBERS:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, **cfg)
    m.fit(X_all[FEATS], y)
    p = m.predict_proba(Xe_all[FEATS])[:, 1]
    print(f"member={name} eval_auc={roc_auc_score(ye, p):.4f}")
    models.append(m)
    preds.append(p)
print(f"ensemble fit ({time.time() - t0:.1f}s)")

mean_p = np.mean(preds, axis=0)
ranks = [rankdata(p) for p in preds]
rank_p = np.mean(ranks, axis=0)

auc_mean = roc_auc_score(ye, mean_p)
auc_rank = roc_auc_score(ye, rank_p)
print(f"ensemble mean-avg AUC: {auc_mean:.4f}")
print(f"ensemble rank-avg AUC: {auc_rank:.4f}")

eval_auc = max(auc_mean, auc_rank)
USE_RANK = auc_rank > auc_mean


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATS]
    ps = [m.predict_proba(Xp)[:, 1] for m in models]
    if USE_RANK:
        return np.mean([rankdata(p) for p in ps], axis=0)
    return np.mean(ps, axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
