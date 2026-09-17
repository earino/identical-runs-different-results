"""XGBoost binary classifier for airline delay. Contract: see program.md.

  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Baseline categorical handling (pd.Categorical with train-fitted levels); DepTime also encoded
as hour/minute numeric; month/day cyclic numeric. Model: XGBoost depth 6, 200 trees, lr 0.1.
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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _num(s):
    return s.str.extract(r"c-(\d+)").astype(float)[0]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    dt = df["DepTime"].fillna(0).astype("int64").clip(0, 2359)
    hour = dt // 100
    minute = dt % 100
    X["DepHour"] = hour.astype(float)
    X["DepMinute"] = minute.astype(float)
    ang = 2 * np.pi * (hour * 60 + minute) / 1440.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    mo = _num(df["Month"]).fillna(1)
    angm = 2 * np.pi * (mo - 1) / 12.0
    X["MonthSin"] = np.sin(angm)
    X["MonthCos"] = np.cos(angm)
    dom = _num(df["DayofMonth"]).fillna(1)
    angd = 2 * np.pi * (dom - 1) / 31.0
    X["DomSin"] = np.sin(angd)
    X["DomCos"] = np.cos(angd)
    dow = _num(df["DayOfWeek"]).fillna(1)
    X["IsWeekend"] = ((dow == 6) | (dow == 7)).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def fit_model(X: pd.DataFrame, y: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    model.fit(X, y)
    return model


TRAIN_CACHE = None


def _train_block():
    global TRAIN_CACHE
    if TRAIN_CACHE is None:
        model = fit_model(prepare(train), to_y(train))
        TRAIN_CACHE = model
    return TRAIN_CACHE


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    model = _train_block()
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
model = _train_block()
print(f"Training time: {time.time() - t0:.1f}s")

t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
