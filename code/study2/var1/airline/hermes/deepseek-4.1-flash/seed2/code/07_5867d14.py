"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Model: a bag of XGBoost classifiers (same feature matrix, different seeds) whose probabilities are averaged.
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
N_BAG = 3

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature definitions --------------------------------------------------------
# `c-<n>` calendar columns are integer valued but stored as strings -> decode to ints.
C_NUM_COLS = ["Month", "DayofMonth", "DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(np.int32) // 100).clip(0, 23)


def _keys(df: pd.DataFrame) -> dict:
    """Categorical keys used for frequency counts and interaction features."""
    h = _hour(df).astype(str)
    return {
        "carrier": df["UniqueCarrier"],
        "origin": df["Origin"],
        "dest": df["Dest"],
        "route": df["Origin"] + "_" + df["Dest"],
        "orig_hour": df["Origin"] + "_" + h,
        "dest_hour": df["Dest"] + "_" + h,
        "carrier_hour": df["UniqueCarrier"] + "_" + h,
        "carrier_origin": df["UniqueCarrier"] + "_" + df["Origin"],
    }


# lookups fitted on TRAINING data only (never on the dataframe passed to prepare)
KEY_TR = _keys(train)
CNT = {k: KEY_TR[k].value_counts() for k in KEY_TR}
COUNT_KEYS = ["carrier", "origin", "dest", "route", "orig_hour", "dest_hour", "carrier_hour"]
CARRIER_HOUR_LEVELS = CNT["carrier_hour"].index


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
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN

    K = _keys(df)
    X["CarrierHour"] = pd.Categorical(K["carrier_hour"], categories=CARRIER_HOUR_LEVELS)

    # scheduled-traffic counts (log) for airport / carrier / route / hour keys
    for k in COUNT_KEYS:
        v = K[k].map(CNT[k])
        X["cnt_" + k] = np.log1p(v.fillna(0).to_numpy()).astype(np.float32)

    # carrier's share of departures at this origin, and a distance x hour interaction
    share = K["carrier_origin"].map(CNT["carrier_origin"]).fillna(0) / K["origin"].map(CNT["origin"]).fillna(1)
    X["carrier_origin_share"] = share.astype(np.float32)
    X["dist_x_hour"] = (X["Distance"] * X["DepHour"]).astype(np.float32)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ---------------------------------------------------------------------
PARAMS = dict(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.02,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_lambda=10.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

X_tr = prepare(train)
y_tr = to_y(train)

models = []
t0 = time.time()
for i in range(N_BAG):
    m = xgb.XGBClassifier(random_state=1000 + 137 * i, **PARAMS)
    m.fit(X_tr, y_tr, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_BAG} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
