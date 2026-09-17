"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here; fitted stats come from module level (train only)."""
    X = pd.DataFrame(index=df.index)
    m = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    deptime = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = deptime // 100
    minute = deptime % 100
    dist = pd.to_numeric(df["Distance"], errors="coerce")

    X["Month"] = m
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["DepTime"] = deptime
    X["Hour"] = hour
    X["Minute"] = minute
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist)
    X["Month_sin"] = np.sin(2 * np.pi * m / 12)
    X["Dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["Dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["Hour_sin"] = np.sin(2 * np.pi * hour / 24)

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


# frequency counts fitted on TRAIN ONLY
_train_cat = prepare(train)
FREQ_MAPS = {c: _train_cat[c].astype(str).value_counts() for c in ["UniqueCarrier", "Origin", "Dest"]}
_train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
ROUTE_FREQ = _train_route.value_counts()


def add_freq(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["Freq_UniqueCarrier"] = df["UniqueCarrier"].astype(str).map(FREQ_MAPS["UniqueCarrier"]).fillna(0).astype(float)
    X["Freq_Origin"] = df["Origin"].astype(str).map(FREQ_MAPS["Origin"]).fillna(0).astype(float)
    X["Freq_Dest"] = df["Dest"].astype(str).map(FREQ_MAPS["Dest"]).fillna(0).astype(float)
    X["Freq_Route"] = route.map(ROUTE_FREQ).fillna(0).astype(float)
    X["Freq_Route_log"] = np.log1p(X["Freq_Route"])
    return X


def features(df: pd.DataFrame) -> pd.DataFrame:
    return add_freq(prepare(df), df)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
params = dict(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model = xgb.XGBClassifier(**params)
t0 = time.time()
model.fit(features(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
