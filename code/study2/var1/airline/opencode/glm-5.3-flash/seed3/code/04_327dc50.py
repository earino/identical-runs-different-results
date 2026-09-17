"""Airline delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

NUM_COLS = ["Month_i", "Day_i", "DoW_i", "DepTime", "DepTime_mod", "DepTime_sin", "DepTime_cos",
            "Distance", "LogDist"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route", "DepHour"]
FEATURE_COLS = NUM_COLS + CAT_COLS


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


CAT_LEVELS = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].astype(str).unique())),
    "Origin": pd.Index(sorted(train["Origin"].astype(str).unique())),
    "Dest": pd.Index(sorted(train["Dest"].astype(str).unique())),
    "Route": pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique())),
    "DepHour": pd.Index([f"h{i:02d}" for i in range(24)]),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month_i"] = _to_int(df["Month"])
    X["Day_i"] = _to_int(df["DayofMonth"])
    X["DoW_i"] = _to_int(df["DayOfWeek"])
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime"] = dt
    dt_mod = dt % 2400
    X["DepTime_mod"] = dt_mod
    hh = (dt_mod // 100).fillna(0).astype(int).clip(0, 23)
    mm = (dt_mod % 100).clip(0, 59)
    minutes = hh * 60 + mm
    X["DepTime_sin"] = np.sin(2 * np.pi * minutes / 1440)
    X["DepTime_cos"] = np.cos(2 * np.pi * minutes / 1440)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["Route"] = X["Origin"] + "_" + X["Dest"]
    X["DepHour"] = "h" + hh.astype(int).astype(str).str.zfill(2)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


MODEL_KW = dict(
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=20,
    reg_lambda=1.0,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    n_estimators=1500,
    n_jobs=N_JOBS,
)

t0 = time.time()
y_all = to_y(train)
X_all = prepare(train)
ENSEMBLE = [
    dict(subsample=0.8, colsample_bytree=0.8, random_state=42),
    dict(subsample=0.7, colsample_bytree=0.9, random_state=7),
    dict(subsample=0.9, colsample_bytree=0.7, random_state=2024),
]
models = []
for kw in ENSEMBLE:
    m = xgb.XGBClassifier(**MODEL_KW, **kw)
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
