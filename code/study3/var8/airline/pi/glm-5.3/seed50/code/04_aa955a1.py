"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Setup: eval (2006) AUC peaks at limited total boost and declines with more capacity (time-shift
overfitting), so: fixed 400 trees, strong feature subsampling, and a 5-seed bagged ensemble.
Features: DepTime decomposition + cyclical tod, log distance, red-eye flag, and training-set
flight-frequency counts (route / origin / dest / carrier) as volume proxies.
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

# --- features (all maps fitted on TRAIN only) -----------------------------------
CAT_COLS = ["Month", "DayOfWeek", "DayofMonth", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cnt_route = _route.value_counts().to_dict()
cnt_origin = train["Origin"].value_counts().to_dict()
cnt_dest = train["Dest"].value_counts().to_dict()
cnt_carrier = train["UniqueCarrier"].value_counts().to_dict()
_tod_train = (train["DepTime"] // 100) * 60 + train["DepTime"] % 100
tod_mean_route = pd.Series(_tod_train).groupby(_route).mean().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    dt = X["DepTime"]
    X["hour"] = dt // 100
    X["minute"] = dt % 100
    X["tod_min"] = X["hour"] * 60 + X["minute"]
    X["tod_sin"] = np.sin(2 * np.pi * X["tod_min"] / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * X["tod_min"] / 1440)
    X["log_dist"] = np.log1p(X["Distance"])
    X["red_eye"] = (X["hour"] < 5).astype(int)
    X["cnt_route"] = np.log1p((df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(cnt_route).fillna(0))
    X["cnt_origin"] = np.log1p(df["Origin"].map(cnt_origin).fillna(0))
    X["cnt_dest"] = np.log1p(df["Dest"].map(cnt_dest).fillna(0))
    X["cnt_carrier"] = np.log1p(df["UniqueCarrier"].map(cnt_carrier).fillna(0))
    _rt = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["tod_dev_route"] = ((X["tod_min"] - _rt.map(tod_mean_route) + 720) % 1440) - 720
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=200,
    max_depth=24,
    learning_rate=0.09,
    min_child_weight=0,
    subsample=0.8,
    colsample_bytree=0.25,
    reg_lambda=1.0,
    max_bin=64,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
N_MODELS = 5

t0 = time.time()
Xtr = prepare(train)
y = to_y(train)
models = []
for s in range(N_MODELS):
    m = xgb.XGBClassifier(**PARAMS, random_state=100 + s)
    m.fit(Xtr, y)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_MODELS} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
