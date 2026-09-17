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

# --- feature engineering ------------------------------------------------------
CN_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["Distance", "DepTimeMin", "DepHour", "DepMinute",
            "MonthNum", "DayNum", "DowNum", "IsWeekend",
            "MonthSin", "MonthCos", "DowSin", "DowCos"]
FEATURE_COLS = NUM_COLS + CAT_COLS
CATEGORIES = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _cn(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    dep_min = (dep // 100) * 60 + (dep % 100)
    dep_min = dep_min.where((dep_min >= 0) & (dep_min < 1440), other=np.nan)
    X["DepTimeMin"] = dep_min
    X["DepHour"] = (dep_min // 60)
    X["DepMinute"] = (dep_min % 60)
    month = _cn(df["Month"])
    dom = _cn(df["DayofMonth"])
    dow = _cn(df["DayOfWeek"])
    X["MonthNum"] = month
    X["DayNum"] = dom
    X["DowNum"] = dow
    X["IsWeekend"] = dow.isin([6, 7]).astype(int)
    X["MonthSin"] = np.sin(2 * np.pi * month / 12)
    X["MonthCos"] = np.cos(2 * np.pi * month / 12)
    X["DowSin"] = np.sin(2 * np.pi * dow / 7)
    X["DowCos"] = np.cos(2 * np.pi * dow / 7)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CATEGORIES[c])
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
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
