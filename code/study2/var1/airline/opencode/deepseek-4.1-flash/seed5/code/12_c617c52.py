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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
FREQ_SPECS = [("Origin",), ("Dest",), ("UniqueCarrier",),
              ("Origin", "DepHour"), ("Dest", "DepHour"), ("UniqueCarrier", "DepHour"),
              ("Origin", "Dest"), ("Origin", "Dest", "DepHour"),
              ("UniqueCarrier", "Origin"), ("UniqueCarrier", "Dest")]
NUM_COLS = ["Distance", "DepTimeMin", "DepHour", "DepMinute",
            "MonthNum", "DayNum", "DowNum", "IsWeekend",
            "MonthSin", "MonthCos", "DowSin", "DowCos"]
FREQ_COLS = [f"freq_{'_'.join(s)}" for s in FREQ_SPECS]
CAT2_SPECS = [("DepHour",), ("UniqueCarrier", "DepHour")]
CAT2_COLS = ["c_DepHour", "c_CarrierHour"]
FEATURE_COLS = NUM_COLS + CAT_COLS + CAT2_COLS + FREQ_COLS


def _key(df: pd.DataFrame, spec) -> pd.Series:
    return df[list(spec)].astype(str).agg("_".join, axis=1)


def _base(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    dep_min = (dep // 100) * 60 + (dep % 100)
    X["DepTimeMin"] = dep_min.where((dep_min >= 0) & (dep_min < 1440), other=np.nan)
    X["DepHour"] = X["DepTimeMin"] // 60
    X["DepMinute"] = X["DepTimeMin"] % 60
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["MonthNum"] = month
    X["DayNum"] = dom
    X["DowNum"] = dow
    X["IsWeekend"] = dow.isin([6, 7]).astype(int)
    X["MonthSin"] = np.sin(2 * np.pi * month / 12)
    X["MonthCos"] = np.cos(2 * np.pi * month / 12)
    X["DowSin"] = np.sin(2 * np.pi * dow / 7)
    X["DowCos"] = np.cos(2 * np.pi * dow / 7)
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    return X


_base_train = _base(train)
CATEGORIES = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
CAT2_CATEGORIES = {spec: pd.Index(sorted(_key(_base_train, spec).dropna().unique())) for spec in CAT2_SPECS}
FREQ_MAPS = {spec: _key(_base_train, spec).value_counts().to_dict() for spec in FREQ_SPECS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    B = _base(df)
    X = B.copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CATEGORIES[c])
    for spec, col in zip(FREQ_SPECS, FREQ_COLS):
        X[col] = _key(B, spec).map(FREQ_MAPS[spec]).fillna(0).astype(float)
    for spec, col in zip(CAT2_SPECS, CAT2_COLS):
        X[col] = pd.Categorical(_key(B, spec), categories=CAT2_CATEGORIES[spec])
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=16,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
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
