"""XGBoost binary classifier for airline delay prediction.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = [
    "Distance", "dep_hour", "dep_min", "dep_sin", "dep_cos", "DepTime_raw",
    "month", "dom", "dow", "is_weekend",
]


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["UniqueCarrier", "Origin", "Dest", "Distance"]].copy()
    dt = df["DepTime"].fillna(0).astype(int)
    X["dep_hour"] = (dt // 100).clip(0, 23)
    X["dep_min"] = X["dep_hour"] * 60 + (dt % 100).clip(0, 59)
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_min"] / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_min"] / 1440.0)
    X["DepTime_raw"] = dt
    X["month"] = _cnum(df["Month"]).fillna(0).astype(int)
    X["dom"] = _cnum(df["DayofMonth"]).fillna(0).astype(int)
    X["dow"] = _cnum(df["DayOfWeek"]).fillna(0).astype(int)
    X["is_weekend"] = X["dow"].isin([6, 7]).astype(int)
    return X


# Categorical levels fit on TRAIN ONLY.
_train_feat = _features(train)
cat_levels = {c: pd.Index(sorted(_train_feat[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba() reproduces it on unseen rows.
    X = _features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[CAT_COLS + NUM_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=350,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.4,
    min_child_weight=5,
    reg_lambda=1.0,
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
