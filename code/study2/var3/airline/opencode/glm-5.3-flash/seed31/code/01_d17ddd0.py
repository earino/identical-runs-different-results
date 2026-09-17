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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_eval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

# --- statistics fit on TRAIN ONLY (applied later inside prepare) ---------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "route", "hour"]
cat_levels = {
    "Month": pd.Index(sorted(train["Month"].astype(str).unique())),
    "DayofMonth": pd.Index(sorted(train["DayofMonth"].astype(str).unique())),
    "DayOfWeek": pd.Index(sorted(train["DayOfWeek"].astype(str).unique())),
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].astype(str).unique())),
    "Origin": pd.Index(sorted(train["Origin"].astype(str).unique())),
    "Dest": pd.Index(sorted(train["Dest"].astype(str).unique())),
    "route": pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique())),
    "hour": pd.Index(sorted((train["DepTime"].astype(int) // 100).unique())),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        if c == "route":
            s = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
        elif c == "hour":
            s = (df["DepTime"].astype(int) // 100).astype(str)
        else:
            s = df[c].astype(str)
        X[c] = pd.Categorical(s, categories=cat_levels[c])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=20,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=75,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train),
    y_train,
    eval_set=[(prepare(evald), y_eval)],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
