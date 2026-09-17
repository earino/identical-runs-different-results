"""Airline delay XGBoost — experiment 2: parse strings to ints, hour/minute features,
native categoricals for carrier/origin/dest, deeper model with early stopping.
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
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str[2:].astype(int)
    X["dom"] = df["DayofMonth"].str[2:].astype(int)
    X["dow"] = df["DayOfWeek"].str[2:].astype(int)
    dt = df["DepTime"]
    X["deptime"] = dt
    X["hour"] = (dt // 100).clip(0, 23)
    X["minute"] = dt % 100
    X["bad_time"] = ((dt % 100) > 59).astype(int)
    dist = df["Distance"].astype(float)
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

t0 = time.time()
X = prepare(train)
y = to_y(train)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
es = xgb.XGBClassifier(early_stopping_rounds=50, n_estimators=800, **PARAMS)
es.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_it = es.best_iteration
print(f"best_iteration={best_it} cv_auc={es.best_score:.4f} es_time={time.time()-t0:.1f}s")

t0 = time.time()
model = xgb.XGBClassifier(n_estimators=best_it + 1, **PARAMS)
model.fit(X, y)
print(f"Training time: {time.time()-t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time()-t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
