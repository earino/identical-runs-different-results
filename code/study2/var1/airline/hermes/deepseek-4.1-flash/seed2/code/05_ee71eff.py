"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features ------------------------------------------------------------------
# `c-<n>` calendar columns are integer valued but stored as strings -> decode to ints.
C_NUM_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(np.int32) // 100).clip(0, 23)


def _keys(df: pd.DataFrame) -> dict:
    """Categorical keys used for both frequency counts and interaction features."""
    h = _hour(df).astype(str)
    return {
        "carrier": df["UniqueCarrier"],
        "origin": df["Origin"],
        "dest": df["Dest"],
        "route": df["Origin"] + "_" + df["Dest"],
        "orig_hour": df["Origin"] + "_" + h,
        "dest_hour": df["Dest"] + "_" + h,
        "carrier_hour": df["UniqueCarrier"] + "_" + h,
    }


# lookups fitted on TRAINING data only (never on the dataframe passed to prepare)
KEY_TR = _keys(train)
CNT = {k: KEY_TR[k].value_counts() for k in KEY_TR}
COUNT_KEYS = ["carrier", "origin", "dest", "route", "orig_hour", "dest_hour", "carrier_hour"]
count_levels = {k: CNT[k].index for k in COUNT_KEYS}
ch_levels = count_levels["carrier_hour"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba() reproduces it on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in C_NUM_COLS:
        X[c] = df[c].str.replace("c-", "", regex=False).astype(np.int16)
    X["Distance"] = df["Distance"].astype(np.float32)

    dep = df["DepTime"].astype(np.int32)
    hour = _hour(df)
    minute = (dep % 100).clip(0, 59)
    X["DepHour"] = hour.astype(np.int16)
    X["DepMinOfDay"] = (hour * 60 + minute).astype(np.int16)

    X["HourSin"] = np.sin(2 * np.pi * X["DepHour"] / 24).astype(np.float32)
    X["HourCos"] = np.cos(2 * np.pi * X["DepHour"] / 24).astype(np.float32)
    doy = (X["Month"] - 1) * 30 + X["DayofMonth"]
    X["DoySin"] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
    X["DoyCos"] = np.cos(2 * np.pi * doy / 365).astype(np.float32)

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["CarrierHour"] = pd.Categorical(_keys(df)["carrier_hour"], categories=ch_levels)

    # scheduled-traffic counts (log) for airport/carrier/route/hour keys
    K = _keys(df)
    for k in COUNT_KEYS:
        v = K[k].map(CNT[k])
        X["cnt_" + k] = np.log1p(v.fillna(0).to_numpy()).astype(np.float32)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ---------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=30,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_lambda=5.0,
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
