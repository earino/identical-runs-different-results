"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
HIGH_CARD = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in HIGH_CARD}


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"c-(\d+)", expand=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dom"] = dom
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    X["dow"] = dow
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    dep = df["DepTime"].astype(float)
    hh = (dep // 100).clip(0, 47)
    mm = dep % 100
    mins = ((hh * 60 + mm) % 1440)
    X["dep_min"] = mins
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["dep_hour"] = mins // 60
    X["distance"] = df["Distance"].astype(float)
    X["distance_log"] = np.log1p(df["Distance"].astype(float))
    for c in HIGH_CARD:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["carrier_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + (mins // 15).astype(int).astype(str),
        categories=[f"{c}_{h}" for c in sorted(cat_levels["UniqueCarrier"]) for h in range(96)],
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.05,
    max_depth=10,
    subsample=0.75,
    colsample_bytree=0.6,
    min_child_weight=20,
    tree_method="hist",
    n_jobs=N_JOBS,
)
N_ROUNDS = 650
N_MODELS = 10

# --- bag: N_MODELS refits on ALL training data, DMatrix built once and shared --
t0 = time.time()
models = []
dfull = xgb.DMatrix(prepare(train), label=to_y(train), enable_categorical=True)
for i in range(N_MODELS):
    models.append(xgb.train({**PARAMS, "seed": SEED + i}, dfull, num_boost_round=N_ROUNDS))
print(f"Training time: {time.time() - t0:.1f}s  n_models={N_MODELS} n_trees={N_ROUNDS}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(prepare(df), enable_categorical=True)
    return np.mean([m.predict(d) for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
