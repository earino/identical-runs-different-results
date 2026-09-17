"""XGBoost binary classifier with engineered calendar/time + interaction features. ONLY FILE THE AGENT EDITS."""
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
# Raw Month/DayofMonth are deliberately excluded: their year-specific seasonality does not transfer,
# while `doy` (within-year position) and the categorical time interactions do.
CAT_SRC = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest", "hour_c"]
DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
DOY_BASE = np.concatenate([[0], np.cumsum(DAYS_IN_MONTH)[:-1]])


def _build(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["Distance"] = df["Distance"].astype(float)

    month = df["Month"].astype(str).str.slice(2).astype(int).to_numpy()
    day = df["DayofMonth"].astype(str).str.slice(2).astype(int).to_numpy()
    X["doy"] = DOY_BASE[month - 1] + day
    X["is_weekend"] = df["DayOfWeek"].isin(["c-6", "c-7"]).astype(int).to_numpy()

    dep = df["DepTime"].astype(float)
    hour = dep // 100
    minute = dep % 100
    ok = (hour <= 23) & (minute <= 59)
    X["hour"] = hour.where(ok, -1).to_numpy()
    X["minute"] = minute.where(ok, -1).to_numpy()
    X["time_frac"] = (hour * 60 + minute).where(ok, -1).to_numpy()
    X["hour_c"] = X["hour"].astype(int).astype(str)

    # interactions: carrier and origin delay behaviour varies strongly by time of day
    X["carrier_hour"] = X["UniqueCarrier"] + "_" + X["hour_c"]
    X["origin_hour"] = X["Origin"] + "_" + X["hour_c"]
    X["carrier_half"] = X["UniqueCarrier"] + "_" + (X["time_frac"] // 30).astype(int).astype(str)
    X["origin_half"] = X["Origin"] + "_" + (X["time_frac"] // 15).astype(int).astype(str)

    # estimated scheduled arrival time (distance / typical cruise speed) captures end-of-day congestion
    arr = X["time_frac"] + X["Distance"] / 7.0
    X["arr_time_frac"] = arr
    X["arr_hour"] = (arr // 60) % 24
    X["arr_hour_c"] = X["arr_hour"].astype(int).astype(str)
    return X


# categorical levels are fit on the training data only
_cat_levels = {}
_tmp = _build(train)
for _c in ["DayOfWeek", "UniqueCarrier", "Origin", "Dest", "hour_c", "carrier_hour", "origin_hour", "carrier_half", "origin_half", "arr_hour_c"]:
    _cat_levels[_c] = pd.Index(sorted(_tmp[_c].dropna().unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _build(df)
    for c, levels in _cat_levels.items():
        X[c] = pd.Categorical(X[c], categories=levels)  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=1500,
    max_depth=9,
    learning_rate=0.02,
    min_child_weight=30,
    reg_lambda=50.0,
    subsample=0.8,
    colsample_bytree=0.5,
    colsample_bynode=0.6,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

X_train = prepare(train)
y_train = to_y(train)

t0 = time.time()
models = [xgb.XGBClassifier(random_state=SEED + i, **PARAMS) for i in range(3)]
for m in models:
    m.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
