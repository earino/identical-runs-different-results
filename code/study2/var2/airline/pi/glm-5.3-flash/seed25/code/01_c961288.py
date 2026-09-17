"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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
RAW_CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT_COLS}
ROUTE_LEVELS = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).dropna().unique()))
HOUR_LEVELS = pd.Index(range(0, 27))


def _int_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # dates as ints and as categories
    m, dom, dow = _int_col(df["Month"]), _int_col(df["DayofMonth"]), _int_col(df["DayOfWeek"])
    X["month"] = m
    X["dayofmonth"] = dom
    X["dayofweek"] = dow
    X["month_cat"] = pd.Categorical(df["Month"], categories=CAT_LEVELS["Month"])
    X["dow_cat"] = pd.Categorical(df["DayOfWeek"], categories=CAT_LEVELS["DayOfWeek"])
    X["dom_cat"] = pd.Categorical(df["DayofMonth"], categories=CAT_LEVELS["DayofMonth"])
    # departure time: hhmm -> hour/minute; hour-of-day is cyclical (24xx-26xx wrap past midnight)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).fillna(0).astype(int)
    minute = (dt % 100).fillna(0).astype(float)
    X["dephour"] = hour
    X["depmin"] = minute
    tod = (hour * 60 + minute) % 1440
    X["tod"] = tod
    ang = 2 * np.pi * tod / 1440.0
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["late_night"] = ((tod < 300) | (tod >= 1380)).astype(int)
    X["hour_cat"] = pd.Categorical(hour.clip(0, 26), categories=HOUR_LEVELS)
    # airports / carrier / route as categories
    X["carrier"] = pd.Categorical(df["UniqueCarrier"], categories=CAT_LEVELS["UniqueCarrier"])
    X["origin"] = pd.Categorical(df["Origin"], categories=CAT_LEVELS["Origin"])
    X["dest"] = pd.Categorical(df["Dest"], categories=CAT_LEVELS["Dest"])
    X["route"] = pd.Categorical(df["Origin"] + "_" + df["Dest"], categories=ROUTE_LEVELS)
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["logdist"] = np.log1p(dist)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
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
