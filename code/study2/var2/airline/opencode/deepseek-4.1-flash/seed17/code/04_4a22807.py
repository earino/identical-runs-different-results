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
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
RAW = [c for c in train.columns if c not in ID_COLS + [TARGET]]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CNT_COLS = ["Origin", "Dest", "UniqueCarrier"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[RAW].copy()
    dep = X["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_time"] = hour * 60 + minute
    X["carr_orig"] = X["UniqueCarrier"].astype(str) + "_" + X["Origin"].astype(str)
    X["carr_dest"] = X["UniqueCarrier"].astype(str) + "_" + X["Dest"].astype(str)
    hh = hour.astype(str)
    X["origin_hour"] = X["Origin"].astype(str) + "_" + hh
    X["dest_hour"] = X["Dest"].astype(str) + "_" + hh
    for c in CNT_COLS:
        X[f"cnt_{c}"] = X[c].map(COUNT_MAPS[c]).fillna(0)
    return X


# frequency maps (fit on training data only → generalize to unseen levels as 0)
_base = pd.DataFrame({c: train[c] for c in CNT_COLS})
COUNT_MAPS = {c: _base[c].value_counts() for c in CNT_COLS}
CAT_ALL = CAT_COLS + ["dep_hour", "carr_orig", "carr_dest", "origin_hour", "dest_hour"]

# fit category levels on training data only
_tmp = add_features(train)
cat_levels = {c: pd.Index(sorted(_tmp[c].dropna().unique())) for c in CAT_ALL}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = add_features(df)
    for c in CAT_ALL:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Ensemble of XGBoost models of varying depth: variance reduction that is robust
# to the year-to-year distribution shift between train (2005) and eval/holdout (2006).
DEPTHS = [5, 6, 7, 8]
models = [
    xgb.XGBClassifier(
        n_estimators=500,
        max_depth=d,
        learning_rate=0.05,
        reg_alpha=5.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    for d in DEPTHS
]

X_train = prepare(train)
y_train = to_y(train)
t0 = time.time()
for m in models:
    m.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


# keep a reference for callers that expect a single `model`
model = models[0]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
