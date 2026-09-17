"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering state (fit on TRAIN only) -----------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
route_levels = pd.Index(
    sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique())
)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["month"] = _num(df["Month"])
    X["day"] = _num(df["DayofMonth"])
    X["dow"] = _num(df["DayOfWeek"])
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 24)
    minute = (dt % 100).clip(0, 59)
    X["dep_time"] = dt
    X["hour"] = hour
    X["minute"] = minute
    tod = hour * 60 + minute
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    X["sin_month"] = np.sin(2 * np.pi * X["month"] / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * X["month"] / 12.0)
    X["distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels
    )
    return X


# --- model --------------------------------------------------------------------
PARAMS = dict(
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
)

t0 = time.time()
X = prepare(train)
y = to_y(train)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
probe = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=60, **PARAMS)
probe.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
best_iter = probe.best_iteration + 1
model = xgb.XGBClassifier(n_estimators=best_iter, **PARAMS)
model.fit(X, y, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={best_iter}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
