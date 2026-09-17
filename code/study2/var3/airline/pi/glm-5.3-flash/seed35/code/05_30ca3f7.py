"""XGBoost binary classifier on airline delays. ONLY FILE THE AGENT EDITS.

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

# --- feature engineering (all inside prepare; encoders fit on train only) ------
RAW_FEATURES = [c for c in train.columns if c not in ID_COLS + [TARGET] and c not in ("Month", "DayofMonth", "DayOfWeek")]


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[RAW_FEATURES].copy()
    month = df["Month"].str[2:].astype(int)
    dom = df["DayofMonth"].str[2:].astype(int)
    dow = df["DayOfWeek"].str[2:].astype(int)
    hour = (X["DepTime"] // 100).clip(0, 26)
    minute = X["DepTime"] % 100
    X["month_n"] = month
    X["dom_n"] = dom
    X["dow_n"] = dow
    X["hour_n"] = hour
    X["minute_n"] = minute
    X["hour_c"] = hour
    X["route"] = X["Origin"] + "_" + X["Dest"]
    X["log_dist"] = np.log1p(X["Distance"])
    return X


_tmp = _base_features(train)
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "hour_c"]
cat_levels = {c: pd.Index(sorted(_tmp[c].dropna().unique())) for c in CAT_COLS}
FEATURE_COLS = [c for c in _tmp.columns if c not in CAT_COLS] + CAT_COLS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model = xgb.XGBClassifier(**PARAMS)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
