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

# --- features -----------------------------------------------------------------
RAW_NUM = ["DepTime", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
STR_NUM = ["Month", "DayofMonth", "DayOfWeek"]  # c-<n> strings -> numeric


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba reproduces it on unseen rows.
    X = pd.DataFrame(index=df.index)
    # parse c-<n> strings into ints
    for c in STR_NUM:
        X[c] = df[c].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    X["Dist_log"] = np.log1p(X["Distance"])
    # scheduled departure hour/minute
    hh = (X["DepTime"] // 100).clip(0, 24)
    mm = X["DepTime"] % 100
    X["DepHour"] = hh
    X["DepMinute"] = mm
    X["DepHourFrac"] = hh + mm / 60.0
    # smooth cyclical versions
    X["sin_hour"] = np.sin(2 * np.pi * X["DepHourFrac"] / 24)
    X["cos_hour"] = np.cos(2 * np.pi * X["DepHourFrac"] / 24)
    X["sin_dow"] = np.sin(2 * np.pi * X["DayOfWeek"] / 7)
    X["cos_dow"] = np.cos(2 * np.pi * X["DayOfWeek"] / 7)
    X["sin_month"] = np.sin(2 * np.pi * X["Month"] / 12)
    X["sin_dom"] = np.sin(2 * np.pi * X["DayofMonth"] / 31)
    # departures in the small hours are the strongest delay signal
    X["Night"] = ((hh <= 5) | (hh >= 21)).astype(int)
    X["Evening"] = ((hh >= 17) & (hh <= 20)).astype(int)
    # 24 x 7 interaction, numeric so no cat split cost
    X["HourxDow"] = X["DepHour"] * 7 + X["DayOfWeek"]
    X["HourxMonth"] = X["DepHour"] * 12 + X["Month"]
    # route features
    X["Dist_bin"] = (X["Distance"] // 250).astype(int)
    # congestion keys (dropped before training): flights per airport-hour in the schedule
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    X["_origin_hour"] = o + "_" + hh.astype(str)
    X["_dest_hour"] = d + "_" + hh.astype(str)
    X["oh_cnt"] = X["_origin_hour"].map(OH_COUNT).astype(float).fillna(0.0)
    X["dh_cnt"] = X["_dest_hour"].map(DH_COUNT).astype(float).fillna(0.0)
    X["origin_cnt"] = o.map(ORIGIN_COUNT).astype(float).fillna(0.0)
    X["dest_cnt"] = d.map(DEST_COUNT).astype(float).fillna(0.0)
    X["route_cnt"] = (o + "_" + d).map(CNT_ROUTE).astype(float).fillna(0.0)
    # finer / wider congestion views
    slot = (hh * 4 + mm // 15).astype(int)
    X["os_cnt"] = (o + "_" + slot.astype(str)).map(OS_COUNT).astype(float).fillna(0.0)
    X["ds_cnt"] = (d + "_" + slot.astype(str)).map(DS_COUNT).astype(float).fillna(0.0)
    X["hour_cnt"] = hh.map(HOUR_COUNT).astype(float).fillna(0.0)
    X["odow_cnt"] = (o + "_" + X["DayOfWeek"].astype(str)).map(ODOW_COUNT).astype(float).fillna(0.0)
    X["ddow_cnt"] = (d + "_" + X["DayOfWeek"].astype(str)).map(DDOW_COUNT).astype(float).fillna(0.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X.drop(columns=["_origin_hour", "_dest_hour"])


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- schedule-congestion statistics fit on TRAIN only ---------------------------
_hour = (train["DepTime"] // 100).clip(0, 24)
_oh = train["Origin"].astype(str) + "_" + _hour.astype(str)
_dh = train["Dest"].astype(str) + "_" + _hour.astype(str)
_rt = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_slot = (train["DepTime"] // 100).clip(0, 24) * 4 + (train["DepTime"] % 100) // 15
OH_COUNT = _oh.value_counts().to_dict()
DH_COUNT = _dh.value_counts().to_dict()
ORIGIN_COUNT = train["Origin"].value_counts().to_dict()
DEST_COUNT = train["Dest"].value_counts().to_dict()
CNT_ROUTE = _rt.value_counts().to_dict()
OS_COUNT = (train["Origin"].astype(str) + "_" + _slot.astype(str)).value_counts().to_dict()
DS_COUNT = (train["Dest"].astype(str) + "_" + _slot.astype(str)).value_counts().to_dict()
HOUR_COUNT = _hour.value_counts().to_dict()
ODOW_COUNT = (train["Origin"].astype(str) + "_" + train["DayOfWeek"].astype(str).str.extract(r"(\d+)", expand=False)).value_counts().to_dict()
DDOW_COUNT = (train["Dest"].astype(str) + "_" + train["DayOfWeek"].astype(str).str.extract(r"(\d+)", expand=False)).value_counts().to_dict()


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of XGBoost models with different seeds/feature fractions: decorrelates trees,
# hedges against the 2005->2006 shift
members = []
Xtr = prepare(train)
ytr = to_y(train)
t0 = time.time()
CFG = [(42, 5, 0.5), (7, 4, 0.4), (2024, 6, 0.6), (11, 5, 0.45), (99, 4, 0.55)]
for seed, depth, cs in CFG:
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=depth,
        learning_rate=0.07,
        subsample=0.7,
        colsample_bytree=cs,
        min_child_weight=10,
        reg_alpha=2.0,
        reg_lambda=10.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    members.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(members)} members)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
