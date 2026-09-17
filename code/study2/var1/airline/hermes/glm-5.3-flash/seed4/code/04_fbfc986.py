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
SEEDS = (42, 7, 13, 101, 202)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- encoders fitted on TRAIN only, applied inside prepare ---------------------
CAT_COLS = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]  # DayofMonth dropped: pure year-overfit noise
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HLEVELS = pd.Index(sorted((train["DepTime"].astype(int) // 100).unique())).astype(str)
CAR_H_LEVELS = pd.Index(
    sorted((train["UniqueCarrier"] + "|" + (train["DepTime"].astype(int) // 100).astype(str)).unique())
)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    dt = df["DepTime"].astype(int)
    hh = (dt // 100).astype(str)
    X = df[["DepTime", "Distance"]].copy()
    X["hcat"] = pd.Categorical(hh, categories=HLEVELS)  # unseen hours -> NaN
    X["hour"] = dt // 100
    X["minute"] = dt % 100
    ang = 2 * np.pi * (dt % 1440) / 1440.0
    X["tsin"] = np.sin(ang)
    X["tcos"] = np.cos(ang)
    X["car_h"] = pd.Categorical(df["UniqueCarrier"] + "|" + hh, categories=CAR_H_LEVELS)
    X["log_dist"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)

PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=16,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.4,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
proba_eval = []
models = []
for seed in SEEDS:
    m = xgb.XGBClassifier(random_state=seed, **PARAMS)
    m.fit(X, y)
    models.append(m)
print(f"Trained {len(SEEDS)} bagged models in {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
