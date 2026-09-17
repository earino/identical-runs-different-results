"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Notes from experimentation:
  - Dominant, year-stable signal: hour of day (DepTime). Fourier terms + minute help shallow trees.
  - Carrier rates transfer 2005->2006 (corr 0.85); airport rates transfer weakly (corr 0.38),
    so target/likelihood encodings of airports HURT transfer -- raw categorical splits are better.
  - Capacity hurts transfer across the year shift: shallow trees (d=3) + many + strong L2 work best.
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


# --- features -----------------------------------------------------------------
cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df["DepTime"] // 100
    mn = df["DepTime"] % 100
    fh = h + mn / 60.0
    F = pd.DataFrame({"minute": mn}, index=df.index)
    for k in (1, 2, 3):
        F[f"sin{k}"] = np.sin(2 * np.pi * k * fh / 24.0)
        F[f"cos{k}"] = np.cos(2 * np.pi * k * fh / 24.0)
    return F


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    return pd.concat([X, _time_features(df)], axis=1)


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.05,
    reg_lambda=10,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
