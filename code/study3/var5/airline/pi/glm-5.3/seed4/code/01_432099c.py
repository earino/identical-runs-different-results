"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature engineering ------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
cat_levels["route"] = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # calendar fields come as c-<n> strings
    X["month"] = df["Month"].astype(str).str[2:].astype(int)
    X["day"] = df["DayofMonth"].astype(str).str[2:].astype(int)
    X["dow"] = df["DayOfWeek"].astype(str).str[2:].astype(int)
    X["doy"] = (X["month"] - 1) * 31 + X["day"]
    # scheduled departure time hhmm -> hour / minute / time-of-day (wraps hour 24/26 -> 0/2)
    dep = df["DepTime"].astype(int)
    hour = (dep // 100) % 24
    minute = dep % 100
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour * 60 + minute
    X["tod_sin"] = np.sin(2 * np.pi * X["tod"] / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * X["tod"] / 1440)
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    # geography / carrier
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"].astype(str), categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"].astype(str), categories=cat_levels["Dest"])
    X["route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=cat_levels["route"]
    )
    X["Distance"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["Distance"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=7,
    learning_rate=0.1,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iteration={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
