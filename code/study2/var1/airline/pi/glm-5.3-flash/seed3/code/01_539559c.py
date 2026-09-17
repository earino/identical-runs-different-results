"""Airline delay XGBoost with engineered features + early stopping on an internal split.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- statistics fitted on TRAIN ONLY (usable inside prepare on any df) ---------
def _to_num(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")

carrier_levels = pd.Index(sorted(train["UniqueCarrier"].astype(str).unique()))
origin_levels = pd.Index(sorted(train["Origin"].astype(str).unique()))
dest_levels = pd.Index(sorted(train["Dest"].astype(str).unique()))
route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
route_levels = pd.Index(sorted(route_tr.unique()))
route_freq = route_tr.value_counts()
origin_freq = train["Origin"].astype(str).value_counts()
dest_freq = train["Dest"].astype(str).value_counts()
carrier_freq = train["UniqueCarrier"].astype(str).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    m = _to_num(df["Month"])
    dom = _to_num(df["DayofMonth"])
    dow = _to_num(df["DayOfWeek"])
    X["month"] = m
    X["dom"] = dom
    X["dow"] = dow
    X["month_sin"] = np.sin(2 * np.pi * m / 12)
    X["month_cos"] = np.cos(2 * np.pi * m / 12)
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = (dt // 100) % 24
    mm = dt % 100
    mins = (hh * 60 + mm) % 1440
    X["deptime_raw"] = dt
    X["min_of_day"] = mins
    X["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["hour4"] = np.floor(mins / 240.0)  # 6 bins of 4h
    X["late_time"] = ((dt > 2359) & (dt < 3000)).astype(float)  # odd 24xx codes

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["dist_bin"] = np.digitize(dist, [250, 500, 750, 1000, 1500, 2000, 3000])

    # holiday-ish flags computable without a year (dow: c-1..c-7, assume 1=Mon .. 7=Sun)
    X["thanksgiving"] = ((m == 11) & (dom >= 22) & (dom <= 28) & (dow == 4)).astype(float)
    X["xmas"] = ((m == 12) & (dom >= 20) & (dom <= 26)).astype(float)
    X["newyear"] = (((m == 12) & (dom >= 29)) | ((m == 1) & (dom <= 2))).astype(float)
    X["july4"] = ((m == 7) & (dom >= 2) & (dom <= 5)).astype(float)
    X["memorial"] = ((m == 5) & (dom >= 25) & (dow == 1)).astype(float)
    X["labor"] = ((m == 9) & (dom <= 7) & (dow == 1)).astype(float)

    X["carrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=carrier_levels)
    X["origin"] = pd.Categorical(df["Origin"].astype(str), categories=origin_levels)
    X["dest"] = pd.Categorical(df["Dest"].astype(str), categories=dest_levels)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route"] = pd.Categorical(route, categories=route_levels)
    X["route_freq"] = np.log1p(route.map(route_freq).fillna(0).astype(float))
    X["origin_freq"] = np.log1p(df["Origin"].astype(str).map(origin_freq).fillna(0).astype(float))
    X["dest_freq"] = np.log1p(df["Dest"].astype(str).map(dest_freq).fillna(0).astype(float))
    X["carrier_freq"] = np.log1p(df["UniqueCarrier"].astype(str).map(carrier_freq).fillna(0).astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=SEED)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
