"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside `prepare(df)`; fitted statistics come from train.csv only.
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

# --- feature engineering (all inside prepare: must apply to the hidden holdout) --
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOUR_LEVELS = pd.Index(sorted((pd.to_numeric(train["DepTime"]) // 100).unique()))


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.split("-").str[-1], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["cat_hour"] = pd.Categorical(X["hour"], categories=HOUR_LEVELS)
    for c in CAT_COLS:
        X[f"cat_{c}"] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


FEATURES = ["DepTime", "Distance", "hour", "minute",
            "cat_Month", "cat_DayofMonth", "cat_DayOfWeek", "cat_UniqueCarrier",
            "cat_Origin", "cat_Dest", "cat_hour"]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.02,
    colsample_bytree=0.6,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(prepare(train)[FEATURES], to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[FEATURES])[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
