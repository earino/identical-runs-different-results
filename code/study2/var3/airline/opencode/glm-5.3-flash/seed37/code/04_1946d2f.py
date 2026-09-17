"""XGBoost binary classifier for airline delay. THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _cnum(s: pd.Series) -> np.ndarray:
    return s.str.slice(2).astype(float).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    dt = df["DepTime"].astype(float)
    hour = np.floor(dt / 100.0)
    minute = dt - hour * 100.0
    tod = (hour * 60.0 + minute) / 1440.0
    X["deptime"] = dt
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)
    dist = df["Distance"].astype(float)
    X["distance"] = dist
    X["distance_log"] = np.log1p(dist)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model_params = dict(
    max_depth=6,
    learning_rate=0.03,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
X_va = prepare(evald)
y_va = to_y(evald)
es_model = xgb.XGBClassifier(**model_params, early_stopping_rounds=100, eval_metric="auc")
es_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_n = int(es_model.best_iteration) + 1
print(f"ES fit: {time.time() - t0:.1f}s, best_iteration={best_n}, eval_auc_during_es={es_model.best_score:.4f}")

t0 = time.time()
model = xgb.XGBClassifier(**{**model_params, "n_estimators": best_n})
model.fit(X_tr, y_tr)
print(f"Refit: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
