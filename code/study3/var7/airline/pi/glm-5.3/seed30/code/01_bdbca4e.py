"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]

# categorical levels learned from TRAIN only (unseen levels in eval/holdout -> NaN)
_route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))
cat_levels = {
    "UniqueCarrier": pd.Index(sorted(train["UniqueCarrier"].dropna().unique())),
    "Origin": pd.Index(sorted(train["Origin"].dropna().unique())),
    "Dest": pd.Index(sorted(train["Dest"].dropna().unique())),
    "Route": _route_levels,
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = (dep // 100) % 24
    minute = dep % 100
    tmin = hour * 60 + minute
    X["DepTime"] = dep
    X["DepTimeMin"] = tmin
    X["DepHour"] = hour
    X["DepMin"] = minute
    X["DepSin"] = np.sin(2 * np.pi * tmin / 1440.0)
    X["DepCos"] = np.cos(2 * np.pi * tmin / 1440.0)
    X["Month"] = df["Month"].astype(str).str.slice(2).astype(int)
    X["DayofMonth"] = df["DayofMonth"].astype(str).str.slice(2).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str).str.slice(2).astype(int)
    X["Distance"] = df["Distance"].astype(float)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"].astype(str), categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"].astype(str), categories=cat_levels["Dest"])
    X["Route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=cat_levels["Route"]
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr_full = prepare(train)
ytr_full = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

BASE_PARAMS = dict(
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# phase 1: find the best iteration count on an internal train/val split
t0 = time.time()
itr, iva = train_test_split(np.arange(len(train)), test_size=0.2, random_state=SEED)
es_model = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, **BASE_PARAMS)
es_model.fit(Xtr_full.iloc[itr], ytr_full[itr], eval_set=[(Xtr_full.iloc[iva], ytr_full[iva])],
             verbose=False)
best_iter = es_model.best_iteration + 1
print(f"ES phase: best_iter={best_iter}  ({time.time() - t0:.1f}s)")

# phase 2: refit on the full training data with the chosen round count
t0 = time.time()
model = xgb.XGBClassifier(n_estimators=best_iter, **BASE_PARAMS)
model.fit(Xtr_full, ytr_full, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
