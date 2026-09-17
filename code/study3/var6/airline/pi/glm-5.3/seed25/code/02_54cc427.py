"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "hour_cat"]


def _num_from_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering: called by prepare(), which predict_proba() uses on unseen rows."""
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24)
    minute = dep % 100
    tod = (hour * 60 + minute) / 1440.0
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod)
    X["tod_cos"] = np.cos(2 * np.pi * tod)

    month = _num_from_c(df["Month"])
    dom = _num_from_c(df["DayofMonth"])
    dow = _num_from_c(df["DayOfWeek"])
    X["month"] = month
    X["dayofmonth"] = dom
    X["dayofweek"] = dow
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31.0)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_distance"] = np.log1p(dist)

    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["hour_cat"] = hour.fillna(-1).astype(int).astype(str)
    return X


# categorical levels fitted on TRAIN ONLY
_feat_train = build_features(train)
cat_levels = {c: pd.Index(sorted(_feat_train[c].unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=3.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
