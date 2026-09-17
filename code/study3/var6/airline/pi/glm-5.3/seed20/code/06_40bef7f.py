"""XGBoost airline-delay classifier: bagged deep-tree ensemble.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame (same columns as train.csv, target may be absent)
     -> 1-D numpy array of P(positive). All feature engineering lives in prepare().
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

N_MEMBERS = 10
PARAMS = dict(max_depth=32, eta=0.1, tree_method="hist", objective="binary:logistic",
              eval_metric="auc", subsample=0.8, colsample_bytree=0.5, min_child_weight=1, lambda_=1)
ROUNDS = 90

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering -------------------------------------------------------
def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7 (NaN-safe)."""
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")

_cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique()))
               for c in ["UniqueCarrier", "Origin", "Dest"]}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow

    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100) % 24
    tod = hour * 60 + dep % 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)

    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c].astype(str), categories=_cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
t0 = time.time()
dtrain = xgb.DMatrix(prepare(train), label=to_y(train), enable_categorical=True)

members = []
for s in range(N_MEMBERS):
    params = dict(PARAMS)
    params["seed"] = SEED + s
    params["lambda"] = params.pop("lambda_")
    members.append(xgb.train(params, dtrain, num_boost_round=ROUNDS, verbose_eval=0))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(prepare(df), enable_categorical=True)
    return np.mean([m.predict(d) for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
