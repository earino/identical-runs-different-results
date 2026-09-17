"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
def _cnum(s: pd.Series) -> pd.Series:
    # values look like "c-4"; strip the prefix and parse the integer
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _dep_parts(df: pd.DataFrame):
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    dep = dep.where(dep > 0)
    hour = np.floor(dep / 100.0)
    minute = dep - hour * 100.0
    hour = hour.where(hour < 24, hour - 24)
    return hour, minute


cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# congestion: how many scheduled flights at this origin/dest/route in this departure hour (train counts)
_tr_hour, _ = _dep_parts(train)
_hk = _tr_hour.fillna(-1).astype(int).astype(str)
cong_maps = {
    "origin_hour_cnt": (train["Origin"].astype(str) + "|" + _hk).value_counts().to_dict(),
    "dest_hour_cnt": (train["Dest"].astype(str) + "|" + _hk).value_counts().to_dict(),
    "route_hour_cnt": (_route(train) + "|" + _hk).value_counts().to_dict(),
    "origin_cnt": train["Origin"].value_counts().to_dict(),
    "dest_cnt": train["Dest"].value_counts().to_dict(),
    "route_cnt": _route(train).value_counts().to_dict(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    hour, minute = _dep_parts(df)
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_minutes"] = hour * 60.0 + minute
    X["dep_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["dep_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["month"] = _cnum(df["Month"])
    X["day"] = _cnum(df["DayofMonth"])
    X["dow"] = _cnum(df["DayOfWeek"])
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7.0)
    X["is_weekend"] = (X["dow"] >= 6).astype(int)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    hk = hour.fillna(-1).astype(int).astype(str)
    route = _route(df)
    X["origin_hour_cnt"] = (df["Origin"].astype(str) + "|" + hk).map(cong_maps["origin_hour_cnt"]).fillna(0)
    X["dest_hour_cnt"] = (df["Dest"].astype(str) + "|" + hk).map(cong_maps["dest_hour_cnt"]).fillna(0)
    X["route_hour_cnt"] = (route + "|" + hk).map(cong_maps["route_hour_cnt"]).fillna(0)
    X["origin_cnt"] = df["Origin"].map(cong_maps["origin_cnt"]).fillna(0)
    X["dest_cnt"] = df["Dest"].map(cong_maps["dest_cnt"]).fillna(0)
    X["route_cnt"] = route.map(cong_maps["route_cnt"]).fillna(0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of xgboost variants --------------------------------------
BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
CONFIGS = [
    dict(max_depth=4, n_estimators=1800, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=5, n_estimators=1200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=6, n_estimators=900, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=6, n_estimators=1200, learning_rate=0.03, subsample=0.7, colsample_bytree=0.6, min_child_weight=20),
    dict(max_depth=7, n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=8, n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10),
    dict(max_depth=9, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=20),
]

X_all = prepare(train)
y_all = to_y(train)
t0 = time.time()
models = []
for i, cfg in enumerate(CONFIGS):
    m = xgb.XGBClassifier(random_state=SEED + i, **{**BASE, **cfg})
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
