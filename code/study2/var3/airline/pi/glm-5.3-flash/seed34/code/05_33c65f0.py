"""Airline delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str.lstrip("c-").astype(float)
    X["DayofMonth"] = df["DayofMonth"].str.lstrip("c-").astype(float)
    X["DayOfWeek"] = df["DayOfWeek"].str.lstrip("c-").astype(float)
    X["Distance"] = df["Distance"].astype(float)
    X["LogDistance"] = np.log1p(X["Distance"])
    X["DistBucket"] = (X["Distance"] // 250).astype(float)
    t = df["DepTime"].fillna(-1).astype(int)
    hour = (t // 100) % 24
    X["DepHour"] = hour
    X["DepMin"] = hour * 60 + t % 100
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=10.0,
    reg_lambda=2.0,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)


def make_model(**kw):
    p = dict(PARAMS)
    p.update(kw)
    return xgb.XGBClassifier(**p)


# base config: CV-selected twice (experiments 11/13): colsample .4, depth 14, ~238 trees
MEMBER_VARIATIONS = [
    {},
    {"random_state": 1337},
    {"random_state": 2024},
    {"random_state": 7},
    {"max_depth": 12, "colsample_bytree": 0.5},
    {"max_depth": 12, "colsample_bytree": 0.35},
    {"learning_rate": 0.03, "n_estimators": 450, "colsample_bytree": 0.45},
    {"max_depth": 10, "min_child_weight": 5.0},
]


t0 = time.time()
X = prepare(train)
y = to_y(train)
models = []
for kw in MEMBER_VARIATIONS:
    cfg = dict(n_estimators=238, max_depth=14, learning_rate=0.05, colsample_bytree=0.4)
    cfg.update(kw)
    m = make_model(**cfg)
    m.fit(X, y)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xd = prepare(df)
    ps = [m.predict_proba(Xd)[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
