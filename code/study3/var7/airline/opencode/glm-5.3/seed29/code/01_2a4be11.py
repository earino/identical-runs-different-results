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

# --- feature engineering --------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
C_PARSE = ["Month", "DayofMonth", "DayOfWeek"]
# levels fit on the TRAINING data only; unseen values in new data -> NaN category
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in C_PARSE:
        X[c] = df[c].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["hour"] = df["DepTime"] // 100
    X["minute"] = df["DepTime"] % 100
    X["dep_minutes"] = X["hour"] * 60 + X["minute"]
    X["Distance"] = df["Distance"]
    X["log_distance"] = np.log1p(df["Distance"].astype(float))
    X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
    callbacks=[xgb.callback.EarlyStopping(rounds=50, maximize=True)],
)

t0 = time.time()
model.fit(
    prepare(train),
    to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
