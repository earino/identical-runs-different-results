"""XGBoost binary classifier on airline delays. Only file the agent edits.

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

FEATURE_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime",
                "UniqueCarrier", "Origin", "Dest", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]


def _num(s):
    # c-<n> -> int
    return s.str[2:].astype(np.int32)


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering. Called on train (module level, for fitting) and inside predict_proba."""
    X = pd.DataFrame(index=df.index)
    month = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    dt = df["DepTime"].astype(np.int32)
    dt = dt.where(dt < 2400, dt - 2400)          # 2400..2411 -> 0..11 (past midnight)
    hour = dt // 100
    minute = dt % 100
    tod = hour * 60 + minute                     # minutes since midnight

    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["hour"] = hour
    X["minute"] = minute
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    X["dom_sin"] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    X["dom_cos"] = np.cos(2 * np.pi * (dom - 1) / 31.0)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    X["dist_log"] = np.log1p(df["Distance"].astype(np.float32))
    X["route"] = df["Origin"].astype(str) + ">" + df["Dest"].astype(str)

    for c in CAT_COLS:
        src = "UniqueCarrier" if c == "UniqueCarrier" else ("Origin" if c == "Origin" else "Dest")
        X[c] = df[c].astype(str) if c != "route" else X["route"]
    return X


# fit category levels on training data only
_fit = _engineer(train)
cat_levels = {c: pd.Index(sorted(_fit[c].unique())) for c in CAT_COLS}
FEATURE_NAMES = [c for c in _fit.columns if c not in CAT_COLS] + CAT_COLS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = _engineer(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[FEATURE_NAMES]


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
