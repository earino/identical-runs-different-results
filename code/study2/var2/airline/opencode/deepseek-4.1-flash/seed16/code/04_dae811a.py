"""XGBoost binary classifier for flight-departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in prepare(), and all fitted statistics are derived from data/train.csv only.
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

NUM_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]

# Frequency maps fit on training data only. Unseen levels map to 0 in prepare().
freq_maps = {}
for c in CAT_COLS:
    freq_maps[c] = train[c].value_counts()
_train_hour = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 23).astype("Int64").astype(str)
freq_maps["route"] = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()
freq_maps["oh"] = (train["Origin"].astype(str) + "_" + _train_hour).value_counts()
freq_maps["dh"] = (train["Dest"].astype(str) + "_" + _train_hour).value_counts()
del _train_hour

# Smoothed target encoding of the two mid-cardinality airport columns, fit on training data only.
_PRIOR = float((train[TARGET] == POSITIVE).mean())
_TARGET_ALPHA = 5.0
tgt_maps = {}
_Y_TRAIN = (train[TARGET] == POSITIVE).astype(int)
for c in ["Origin", "Dest"]:
    g = _Y_TRAIN.groupby(train[c]).agg(["sum", "count"])
    tgt_maps[c] = (g["sum"] + _PRIOR * _TARGET_ALPHA) / (g["count"] + _TARGET_ALPHA)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    hour_s = hour.astype("Int64").astype(str)
    out = pd.DataFrame(index=df.index)
    out["DepTime"] = dep
    out["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    out["hour"] = hour
    out["minute"] = dep % 100
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    route = origin + "_" + dest
    for c in CAT_COLS:
        out[c + "_freq"] = df[c].map(freq_maps[c]).fillna(0).astype(np.float32)
    out["route_freq"] = route.map(freq_maps["route"]).fillna(0).astype(np.float32)
    out["oh_freq"] = (origin + "_" + hour_s).map(freq_maps["oh"]).fillna(0).astype(np.float32)
    out["dh_freq"] = (dest + "_" + hour_s).map(freq_maps["dh"]).fillna(0).astype(np.float32)
    for c in ["Origin", "Dest"]:
        out[c + "_tgt"] = df[c].map(tgt_maps[c]).fillna(_PRIOR).astype(np.float32)
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=10,
    learning_rate=0.02,
    min_child_weight=5,
    reg_lambda=5.0,
    subsample=0.9,
    colsample_bytree=0.8,
    tree_method="hist",
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
