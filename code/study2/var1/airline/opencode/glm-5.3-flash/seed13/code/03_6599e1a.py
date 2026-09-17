"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepHour", "CarrierDepQ"]
NUM_COLS = ["Distance", "LogDist", "DepMinSin", "DepMinCos"]


def _to_int(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _to_int(df["Month"])
    X["DayofMonth"] = _to_int(df["DayofMonth"])
    X["DayOfWeek"] = _to_int(df["DayOfWeek"])
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).astype(float)
    minute = (dt % 100).astype(float)
    hour = hour.where(hour < 24, hour - 24)  # 2400-26xx quirks
    dep_min = (hour * 60 + minute).clip(0, 24 * 60 - 1)
    X["DepHour"] = (dep_min // 60).fillna(-1).astype(int)
    dep_q = (dep_min // 15).fillna(-1).astype(int)
    X["CarrierDepHour"] = X["UniqueCarrier"] + "_" + X["DepHour"].astype(str)
    X["CarrierDepQ"] = X["UniqueCarrier"] + "_" + dep_q.astype(str)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    ang = 2 * np.pi * dep_min / 1440.0
    X["DepMinSin"] = np.sin(ang)
    X["DepMinCos"] = np.cos(ang)
    return X


_tr_feat = build_features(train)
cat_levels = {c: pd.Index(sorted(_tr_feat[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest", "CarrierDepQ"]}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = build_features(df)
    for c in ["UniqueCarrier", "Origin", "Dest", "CarrierDepQ"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in ["Month", "DayofMonth", "DayOfWeek", "DepHour"]:
        X[c] = pd.Categorical(X[c].fillna(-1).astype(int).astype(str))
    return X[CAT_COLS + NUM_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
FINAL_K = 500

t0 = time.time()
X = prepare(train)
y = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

model = xgb.XGBClassifier(**PARAMS, n_estimators=FINAL_K, eval_metric="auc")
model.fit(X, y, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
