"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives inside `prepare()` (which predict_proba calls on unseen rows); every statistic it
uses is fitted once on the training frame and stored in module-level artifacts.
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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- feature engineering (statistics fitted on train only) ---------------------
CAT_SRC = ["UniqueCarrier", "Origin", "Dest"]

FEATS = [
    "MonthNum", "DayofMonthNum", "DayOfWeekNum",
    "MonthSin", "MonthCos", "DoWSin", "DoWCos",
    "Hour", "Minute", "HourFrac", "IsRedEye",
    "Distance", "LogDistance",
    "UniqueCarrier", "Origin", "Dest",
]


def _strip_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


_cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_SRC}

MONTH_ANG = 2 * np.pi / 12.0
DOW_ANG = 2 * np.pi / 7.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month, day, dow = _strip_c(df["Month"]), _strip_c(df["DayofMonth"]), _strip_c(df["DayOfWeek"])
    X["MonthNum"] = month
    X["DayofMonthNum"] = day
    X["DayOfWeekNum"] = dow
    X["MonthSin"] = np.sin(month * MONTH_ANG)
    X["MonthCos"] = np.cos(month * MONTH_ANG)
    X["DoWSin"] = np.sin(dow * DOW_ANG)
    X["DoWCos"] = np.cos(dow * DOW_ANG)

    dept = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dept // 100.0).clip(0, 24)
    minute = (dept % 100.0).clip(0, 59)
    hf = hour + minute / 60.0
    X["Hour"] = hour
    X["Minute"] = minute
    X["HourFrac"] = hf
    X["IsRedEye"] = (hf < 6.0).astype(np.float32)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist.clip(lower=0))

    for c in CAT_SRC:
        X[c] = pd.Categorical(df[c].astype(str), categories=_cat_levels[c])
    return X[FEATS]


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
