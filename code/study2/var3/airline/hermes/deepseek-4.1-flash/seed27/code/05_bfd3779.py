"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     Everything the model consumes is produced inside `prepare()`, and every statistic/encoder is fit on the
     TRAINING rows only, so `predict_proba` reproduces the same transformation on unseen rows.
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
CAL_COLS = ["Month", "DayofMonth", "DayOfWeek"]      # stored as 'c-<n>' strings
FEATURE_COLS = [c for c in TASK["columns"] if c not in ID_COLS + [TARGET]]
CAT_COLS = [c for c in FEATURE_COLS if c not in CAL_COLS + ["DepTime", "Distance"]]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def cnum(s: pd.Series) -> pd.Series:
    """'c-12' -> 12."""
    return s.astype(str).str.slice(2).astype(int)


def dep_hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(int) // 100) % 24


# Schedule-density ("congestion") statistics: how many flights share the same airport/route at the same
# scheduled hour. These are pure counts from the TRAINING data (no target involved), so they carry over to
# later periods, where they act as a proxy for how busy the airport/route is at that hour.
_h = dep_hour(train)
CNT = {
    "orig_hour": (train["Origin"] + "|" + _h.astype(str)).value_counts(),
    "dest_hour": (train["Dest"] + "|" + _h.astype(str)).value_counts(),
    "route_hour": (train["Origin"] + "_" + train["Dest"] + "|" + _h.astype(str)).value_counts(),
    "origin": train["Origin"].value_counts(),
    "dest": train["Dest"].value_counts(),
    "carrier_origin": (train["UniqueCarrier"] + "|" + train["Origin"]).value_counts(),
    "carrier_hour": (train["UniqueCarrier"] + "|" + _h.astype(str)).value_counts(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CAL_COLS:
        X[c] = cnum(df[c])
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    X["hour"] = (dep // 100) % 24
    X["minute"] = dep % 100
    X["dep_min"] = X["hour"] * 60 + X["minute"]
    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    h = X["hour"].astype(str)
    oh = (df["Origin"] + "|" + h).map(CNT["orig_hour"]).fillna(0.0).astype(float)
    dh = (df["Dest"] + "|" + h).map(CNT["dest_hour"]).fillna(0.0).astype(float)
    rh = (df["Origin"] + "_" + df["Dest"] + "|" + h).map(CNT["route_hour"]).fillna(0.0).astype(float)
    oc = df["Origin"].map(CNT["origin"]).fillna(0.0).astype(float)
    dc = df["Dest"].map(CNT["dest"]).fillna(0.0).astype(float)
    X["orig_hour_cnt"] = oh
    X["dest_hour_cnt"] = dh
    X["route_hour_cnt"] = rh
    X["orig_cnt"] = oc
    X["dest_cnt"] = dc
    X["orig_hour_share"] = oh / oc.replace(0.0, 1.0)
    X["carrier_origin_cnt"] = (df["UniqueCarrier"] + "|" + df["Origin"]).map(CNT["carrier_origin"]).fillna(0.0).astype(float)
    X["carrier_hour_cnt"] = (df["UniqueCarrier"] + "|" + h).map(CNT["carrier_hour"]).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=20,
    learning_rate=0.01,
    min_child_weight=1,
    reg_lambda=5.0,
    subsample=0.7,
    colsample_bytree=0.4,
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
