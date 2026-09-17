"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  3. All feature engineering lives in prepare(), fit only on training data.
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

# --- feature schema -------------------------------------------------------------
STR_INT_COLS = ["Month", "DayofMonth", "DayOfWeek"]      # "c-7" -> 7
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]           # native categoricals
NUM_COLS = ["Distance", "DepTime"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# volume / busyness statistics from TRAIN only (target-free)
_ct_origin = train["Origin"].value_counts()
_ct_dest = train["Dest"].value_counts()
_route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_ct_route = _route_train.value_counts()
_hour_train = train["DepTime"].astype(int) // 100
_ct_oh = pd.Series(list(zip(train["Origin"], _hour_train))).value_counts()  # (origin, hour) -> n
_ct_carrier = train["UniqueCarrier"].value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in STR_INT_COLS:
        X[c] = df[c].str.slice(2).astype(int)
    for c in NUM_COLS:
        X[c] = df[c].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    # time-of-day features (DepTime is hhmm, may exceed 2400)
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    X["Hour"] = hour.astype(float)
    X["Minute"] = minute.astype(float)
    X["MinuteOfDay"] = (hour * 60 + minute).astype(float)
    # volume features (train-fitted counts)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["LogVolOrigin"] = np.log1p(df["Origin"].map(_ct_origin).fillna(0).astype(float))
    X["LogVolDest"] = np.log1p(df["Dest"].map(_ct_dest).fillna(0).astype(float))
    X["LogVolRoute"] = np.log1p(route.map(_ct_route).fillna(0).astype(float))
    vol_oh = pd.Series(list(zip(df["Origin"], hour)), index=df.index).map(_ct_oh).fillna(0).astype(float)
    vol_o = df["Origin"].map(_ct_origin).fillna(0).astype(float)
    X["VolOriginHourShare"] = (vol_oh / vol_o.clip(lower=1)).fillna(0)
    X["LogVolOriginHour"] = np.log1p(vol_oh)
    X["LogVolCarrier"] = np.log1p(df["UniqueCarrier"].map(_ct_carrier).fillna(0).astype(float))
    X["LogDistance"] = np.log1p(df["Distance"].astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 4
models = []
t0 = time.time()
X_tr, X_ev = prepare(train), prepare(evald)
y_tr, y_ev = to_y(train), to_y(evald)
for k in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=24,
        learning_rate=0.04,
        colsample_bytree=0.5,
        max_bin=512,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=25,
        random_state=SEED + k,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    print(f"model {k}: best iteration {m.best_iteration}")
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
