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

# --- feature specification ----------------------------------------------------
# raw string columns used as categoricals (low cardinality, transfer across years)
CAT_RAW = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype(float)
    hour = np.clip(dep // 100, 0, 23)
    minute = np.clip(dep % 100, 0, 59)
    frac = hour + minute / 60.0
    X["DepTime_hour"] = hour
    X["DepTime_minute"] = minute
    X["DepTime_frac"] = frac
    X["tod_sin"] = np.sin(2 * np.pi * frac / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * frac / 24.0)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)

    def catnum(s):
        return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")

    month = catnum(df["Month"])
    day = catnum(df["DayofMonth"])
    dow = catnum(df["DayOfWeek"])
    doy = (month - 1) * 30.0 + day
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 360.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 360.0)
    X["is_weekend"] = (dow >= 6).astype(float)
    X["dep_bucket"] = (hour // 3).astype(float)
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all = prepare(train)
y_all = to_y(train)

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.4,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_all, y_all, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
