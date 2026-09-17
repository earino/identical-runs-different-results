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
# Low-cardinality columns use an ordinal code (c-1..c-N ordering); high-cardinality ones use frequency.
ORD_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier"]

# Frequency maps fit on training data only. Unseen levels map to 0 in prepare().
freq_maps = {}
for c in CAT_COLS:
    freq_maps[c] = train[c].value_counts()
ord_maps = {c: {v: i for i, v in enumerate(sorted(train[c].unique()))} for c in ORD_COLS}
_train_hour = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 23).astype("Int64").astype(str)
freq_maps["route"] = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()
freq_maps["oh"] = (train["Origin"].astype(str) + "_" + _train_hour).value_counts()
freq_maps["dh"] = (train["Dest"].astype(str) + "_" + _train_hour).value_counts()
freq_maps["co"] = (train["UniqueCarrier"].astype(str) + "_" + train["Origin"].astype(str)).value_counts()
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
        if c in ORD_COLS:
            out[c + "_ord"] = df[c].map(ord_maps[c]).fillna(-1).astype(np.float32)
        else:
            out[c + "_freq"] = df[c].map(freq_maps[c]).fillna(0).astype(np.float32)
    out["route_freq"] = route.map(freq_maps["route"]).fillna(0).astype(np.float32)
    out["oh_freq"] = (origin + "_" + hour_s).map(freq_maps["oh"]).fillna(0).astype(np.float32)
    out["dh_freq"] = (dest + "_" + hour_s).map(freq_maps["dh"]).fillna(0).astype(np.float32)
    out["oh_share"] = out["oh_freq"] / (out["Origin_freq"] + 1.0)
    out["dh_share"] = out["dh_freq"] / (out["Dest_freq"] + 1.0)
    carrier_freq = df["UniqueCarrier"].map(freq_maps["UniqueCarrier"]).fillna(0).astype(np.float32)
    co_freq = (df["UniqueCarrier"].astype(str) + "_" + origin).map(freq_maps["co"]).fillna(0).astype(np.float32)
    out["carrier_freq"] = carrier_freq
    out["co_freq"] = co_freq
    out["co_share_c"] = co_freq / (carrier_freq + 1.0)
    out["co_share_o"] = co_freq / (out["Origin_freq"] + 1.0)
    for c in ["Origin", "Dest"]:
        out[c + "_tgt"] = df[c].map(tgt_maps[c]).fillna(_PRIOR).astype(np.float32)
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_train = prepare(train)
y_train = to_y(train)
dmat = xgb.DMatrix(X_train, label=y_train)

# Delay probability rises monotonically with scheduled departure hour; encode that as a constraint.
_feature_order = list(X_train.columns)
MONO = tuple(1 if c == "hour" else 0 for c in _feature_order)

PARAMS = dict(
    objective="reg:squarederror",  # rank-based AUC: squared error generalizes better here than logistic
    max_depth=10,
    eta=0.02,
    min_child_weight=5,
    reg_lambda=5.0,
    subsample=0.9,
    colsample_bytree=0.6,
    monotone_constraints=MONO,
    tree_method="hist",
)
N_ROUNDS = 1200
SEEDS = [1, 2, 3, 4, 5]

t0 = time.time()
boosters = []
for s in SEEDS:
    p = dict(PARAMS, seed=s, nthread=N_JOBS)
    boosters.append(xgb.train(p, dmat, num_boost_round=N_ROUNDS))
print(f"Training time: {time.time() - t0:.1f}s  ({len(boosters)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = xgb.DMatrix(prepare(df))
    p = np.mean([b.predict(X) for b in boosters], axis=0)
    return np.clip(p, 0.0, 1.0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
