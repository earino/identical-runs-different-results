"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
NUM_COLS = [c for c in feature_cols if c not in cat_cols]  # DepTime, Distance
# calendar string columns get numeric versions instead; carriers/airports get one-hot
CAL = {"Month": "Month_n", "DayofMonth": "DayofMonth_n", "DayOfWeek": "DayOfWeek_n"}
ONEHOT_COLS = [c for c in cat_cols if c not in CAL]
onehot_levels = {c: sorted(train[c].dropna().unique()) for c in ONEHOT_COLS}


def key_route(df):
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


# structural summaries computed on TRAIN only: mean flight distance per origin/dest/route
_global_mean_dist = float(train["Distance"].mean())
_dist_maps = {
    "origin_avg_dist": train.groupby("Origin")["Distance"].mean().to_dict(),
    "dest_avg_dist": train.groupby("Dest")["Distance"].mean().to_dict(),
    "route_avg_dist": train.assign(k=key_route(train)).groupby("k")["Distance"].mean().to_dict(),
}
# carrier x hour interaction dummies for well-supported cells (schedule-structural, repeats yearly)
_ch_keys = train["UniqueCarrier"].astype(str) + "_" + (train["DepTime"] // 100).astype(int).astype(str)
_ch_counts = _ch_keys.value_counts()
big_carrier_hours = sorted(_ch_counts[_ch_counts >= 100].index)
_oh_keys = train["Origin"].astype(str) + "_" + (train["DepTime"] // 100).astype(int).astype(str)
_oh_counts = _oh_keys.value_counts()
big_origin_hours = sorted(_oh_counts[_oh_counts >= 100].index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in NUM_COLS:
        X[c] = df[c].astype(float)
    for c, name in CAL.items():
        X[name] = df[c].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    X["Hour"] = dep // 100
    X["MinuteOfDay"] = dep // 100 * 60 + dep % 100
    X["SinHour"] = np.sin(2 * np.pi * X["MinuteOfDay"] / 1440.0)
    X["CosHour"] = np.cos(2 * np.pi * X["MinuteOfDay"] / 1440.0)
    X["origin_avg_dist"] = df["Origin"].astype(str).map(_dist_maps["origin_avg_dist"]).fillna(_global_mean_dist)
    X["dest_avg_dist"] = df["Dest"].astype(str).map(_dist_maps["dest_avg_dist"]).fillna(_global_mean_dist)
    X["route_avg_dist"] = key_route(df).map(_dist_maps["route_avg_dist"]).fillna(_global_mean_dist)
    ch = df["UniqueCarrier"].astype(str) + "_" + (df["DepTime"].astype(int) // 100).astype(int).astype(str)
    for kh in big_carrier_hours:
        X[f"ch_{kh}"] = (ch == kh).astype(np.int8)
    oh = df["Origin"].astype(str) + "_" + (df["DepTime"].astype(int) // 100).astype(int).astype(str)
    for kh in big_origin_hours:
        X[f"oh_{kh}"] = (oh == kh).astype(np.int8)
    for c in ONEHOT_COLS:
        v = df[c].astype(str)
        for lev in onehot_levels[c]:
            X[f"{c}_{lev}"] = (v == lev).astype(np.int8)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10,
    tree_method="hist",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
# recency weighting: flights closer to the end of 2005 are closer to the 2006 distribution
w = 0.7 + 0.3 * (train["Month"].str.slice(2).astype(int) - 1) / 11.0
model.fit(prepare(train), to_y(train), sample_weight=w.to_numpy())
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
