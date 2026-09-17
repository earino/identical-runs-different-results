"""Airline delay XGBoost classifier — experiment: feature engineering v1 (no target encoding).

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "hour", "minute_of_day", "hour_sin", "hour_cos",
            "month_sin", "month_cos", "dow_sin", "dow_cos",
            "dom_num", "dow_num", "month_num", "Distance", "log_dist"]

# cat_levels fitted on TRAIN only; unseen levels -> NaN
def _add_route(df):
    return df

cat_levels = {}
for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
    cat_levels[c] = pd.Index(sorted(train[c].dropna().unique()))
_rt = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str))
cat_levels["Route"] = pd.Index(sorted(_rt.unique()))

DIST_BINS = [-1, 200, 400, 700, 1000, 1500, 2500, 10**6]
DIST_LABELS = ["<=200", "201-400", "401-700", "701-1000", "1001-1500", "1501-2500", "2500+"]
HOUR_CATS = [str(i) for i in range(24)]


def _time_parts(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["DepTime"].fillna(0).astype(int)
    hh = (dt // 100) % 24
    mm = dt % 100
    return hh, mm


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    n = len(df)
    X = pd.DataFrame(index=df.index)
    hh, mm = _time_parts(df)
    hour = hh.astype(float)
    mod = hour * 60.0 + mm  # minute of day (24-26h wraps via %24)
    dom = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(float)
    dow = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(float)
    mon = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(float)

    X["DepTime"] = df["DepTime"].astype(float)
    X["hour"] = hour
    X["minute_of_day"] = mod
    X["hour_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    X["hour_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    X["month_sin"] = np.sin(2 * np.pi * (mon - 1) / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * (mon - 1) / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    X["dom_num"] = dom
    X["dow_num"] = dow
    X["month_num"] = mon
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].clip(lower=0))

    # categoricals with train-fitted levels
    for c, src in [("Month", df["Month"]), ("DayofMonth", df["DayofMonth"]),
                   ("DayOfWeek", df["DayOfWeek"]), ("UniqueCarrier", df["UniqueCarrier"]),
                   ("Origin", df["Origin"]), ("Dest", df["Dest"])]:
        X[c] = pd.Categorical(src, categories=cat_levels[c])

    return X[CAT_COLS + NUM_COLS]


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
