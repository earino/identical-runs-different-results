"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature engineering (all inside prepare()) --------------------------------
cat_levels = {
    "Month": pd.Index(sorted(train["Month"].dropna().unique())),
    "DayofMonth": pd.Index(sorted(train["DayofMonth"].dropna().unique())),
    "DayOfWeek": pd.Index(sorted(train["DayOfWeek"].dropna().unique())),
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Distance"] = df["Distance"].astype(float)
    # DepTime: hour and minute as a true clock, cyclical (24xx/25xx/26xx are late-night red-eyes -> %24)
    hour = (df["DepTime"] // 100) % 24
    minute = df["DepTime"] % 100
    ang = 2 * np.pi * (hour * 60 + minute) / 1440
    X["dt_sin"] = np.sin(ang)
    X["dt_cos"] = np.cos(ang)
    X["DepTime"] = df["DepTime"].astype(float)
    # categoricals (unseen levels -> NaN, handled by XGBoost)
    for c, levels in cat_levels.items():
        X[c] = pd.Categorical(df[c], categories=levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
DTRAIN = xgb.DMatrix(prepare(train), label=to_y(train), enable_categorical=True)
DVAL = xgb.DMatrix(prepare(evald), label=to_y(evald), enable_categorical=True)

# --- model ---------------------------------------------------------------------
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 10,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "nthread": N_JOBS,
}
EVALS = [(DTRAIN, "train"), (DVAL, "eval")]

model = xgb.train(
    PARAMS,
    DTRAIN,
    num_boost_round=800,
    evals=EVALS,
    early_stopping_rounds=50,
    verbose_eval=50,
)

t0 = time.time()


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dm = xgb.DMatrix(prepare(df), enable_categorical=True)
    return model.predict(dm, iteration_range=(0, model.best_iteration + 1))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
