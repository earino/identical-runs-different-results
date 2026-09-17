"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
obj_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]
            and (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _strip_cat(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _strip_cat(df["Month"])
    dom = _strip_cat(df["DayofMonth"])
    dow = _strip_cat(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    tod = hour * 60 + minute
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["dep_time"] = dep
    X["dep_hour"] = hour
    X["dep_tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    return X


NUM_COLS = list(_add_features(train).columns)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _add_features(df)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[NUM_COLS + cat_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
from sklearn.model_selection import train_test_split

model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=30,
    random_state=SEED,
    n_jobs=N_JOBS,
)

Xtr, Xva, ytr, yva = train_test_split(
    prepare(train), to_y(train), test_size=0.1, random_state=SEED, stratify=to_y(train)
)
t0 = time.time()
model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
