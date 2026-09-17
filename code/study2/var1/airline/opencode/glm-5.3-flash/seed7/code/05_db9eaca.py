"""XGBoost binary classifier for airline delays. Only file the agent edits.

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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _int_from_c(s: pd.Series) -> pd.Series:
    return s.str[2:].astype(np.int32)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month, dom, dow = df["Month"], df["DayofMonth"], df["DayOfWeek"]
    if month.dtype == object:
        month = _int_from_c(month)
    if dom.dtype == object:
        dom = _int_from_c(dom)
    if dow.dtype == object:
        dow = _int_from_c(dow)
    X["month"] = month
    X["dayofmonth"] = dom
    X["dayofweek"] = dow
    deptime = df["DepTime"].astype(np.int32)
    X["deptime"] = deptime
    X["dep_hour"] = deptime // 100
    X["dep_min"] = deptime % 100
    X["distance"] = df["Distance"].astype(np.float32)
    X["log_distance"] = np.log1p(X["distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


# --- model --------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)
eval_X, eval_y = prepare(evald), to_y(evald)

t0 = time.time()
models = []
for seed in (42, 7, 123, 2024):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=16,
        subsample=0.7,
        colsample_bytree=0.6,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=25,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X_all, y_all, eval_set=[(eval_X, eval_y)], verbose=False)
    print(f"seed {seed}: best_iter={m.best_iteration}")
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
