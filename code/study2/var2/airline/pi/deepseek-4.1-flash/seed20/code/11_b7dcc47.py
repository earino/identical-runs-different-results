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
from sklearn.model_selection import train_test_split

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
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _hour_of(dt: pd.Series) -> pd.Series:
    return (dt.fillna(0).astype(int) // 100).clip(0, 23)


tr_hour = _hour_of(train["DepTime"])
cnt_origin_hour = (train["Origin"].astype(str) + "_" + tr_hour.astype(str)).value_counts()
cnt_carrier_hour = (train["UniqueCarrier"].astype(str) + "_" + tr_hour.astype(str)).value_counts()
cnt_dest_hour = (train["Dest"].astype(str) + "_" + tr_hour.astype(str)).value_counts()
_tr_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cnt_route_hour = (_tr_route + "_" + tr_hour.astype(str)).value_counts()
cnt_origin = train["Origin"].astype(str).value_counts()
cnt_dest = train["Dest"].astype(str).value_counts()
cnt_carrier = train["UniqueCarrier"].astype(str).value_counts()
cnt_route = _tr_route.value_counts()
tr_dow = train["DayOfWeek"].astype(str)
cnt_origin_dow = (train["Origin"].astype(str) + "_" + tr_dow).value_counts()
cnt_dest_dow = (train["Dest"].astype(str) + "_" + tr_dow).value_counts()
cnt_carrier_dow = (train["UniqueCarrier"].astype(str) + "_" + tr_dow).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    dt = X["DepTime"].fillna(0).astype(int)
    hour = (dt // 100).clip(0, 23)
    minute = (dt % 100).clip(0, 59)
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour * 60 + minute
    X["origin_hour_cnt"] = (
        (df["Origin"].astype(str) + "_" + hour.astype(str)).map(cnt_origin_hour).fillna(0).astype(float)
    )
    X["dest_hour_cnt"] = (
        (df["Dest"].astype(str) + "_" + hour.astype(str)).map(cnt_dest_hour).fillna(0).astype(float)
    )
    X["carrier_hour_cnt"] = (
        (df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)).map(cnt_carrier_hour).fillna(0).astype(float)
    )
    route_h = df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + hour.astype(str)
    X["route_hour_cnt"] = route_h.map(cnt_route_hour).fillna(0).astype(float)
    X["origin_hour_frac"] = X["origin_hour_cnt"] / df["Origin"].astype(str).map(cnt_origin).fillna(1).astype(float)
    X["dest_hour_frac"] = X["dest_hour_cnt"] / df["Dest"].astype(str).map(cnt_dest).fillna(1).astype(float)
    X["carrier_hour_frac"] = X["carrier_hour_cnt"] / df["UniqueCarrier"].astype(str).map(cnt_carrier).fillna(1).astype(float)
    X["route_cnt"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(cnt_route).fillna(0).astype(float)
    X["route_hour_frac"] = X["route_hour_cnt"] / X["route_cnt"].replace(0, 1)
    prev_hour = (hour - 1).clip(0, 23)
    X["origin_prev_hour_cnt"] = (
        (df["Origin"].astype(str) + "_" + prev_hour.astype(str)).map(cnt_origin_hour).fillna(0).astype(float)
    )
    X["dest_prev_hour_cnt"] = (
        (df["Dest"].astype(str) + "_" + prev_hour.astype(str)).map(cnt_dest_hour).fillna(0).astype(float)
    )
    dow_s = df["DayOfWeek"].astype(str)
    X["origin_dow_cnt"] = (df["Origin"].astype(str) + "_" + dow_s).map(cnt_origin_dow).fillna(0).astype(float)
    X["dest_dow_cnt"] = (df["Dest"].astype(str) + "_" + dow_s).map(cnt_dest_dow).fillna(0).astype(float)
    X["carrier_dow_cnt"] = (
        df["UniqueCarrier"].astype(str) + "_" + dow_s
    ).map(cnt_carrier_dow).fillna(0).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=9, colsample_bytree=0.5, min_child_weight=20, reg_lambda=5.0, n_estimators=100, learning_rate=0.02),
    dict(max_depth=10, colsample_bytree=0.6, min_child_weight=40, reg_lambda=10.0, n_estimators=100, learning_rate=0.02),
    dict(max_depth=11, colsample_bytree=0.6, min_child_weight=80, reg_lambda=20.0, n_estimators=90, learning_rate=0.02),
    dict(max_depth=8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5.0, n_estimators=120, learning_rate=0.02),
    dict(max_depth=10, colsample_bytree=0.4, min_child_weight=40, reg_lambda=10.0, n_estimators=110, learning_rate=0.02),
    dict(max_depth=10, colsample_bytree=0.6, min_child_weight=40, reg_lambda=10.0, n_estimators=45, learning_rate=0.05),
    dict(max_depth=9, colsample_bytree=0.5, min_child_weight=20, reg_lambda=5.0, n_estimators=70, learning_rate=0.03),
    dict(max_depth=8, colsample_bytree=0.7, min_child_weight=20, reg_lambda=5.0, n_estimators=60, learning_rate=0.04),
]

Xtr, Xva, ytr, yva = train_test_split(
    prepare(train), to_y(train), test_size=0.15, random_state=SEED, stratify=to_y(train)
)
Xfull, yfull = prepare(train), to_y(train)

t0 = time.time()
models = []
for i, cfg in enumerate(CONFIGS):
    for rep in range(2):
        m = xgb.XGBClassifier(
            subsample=0.8,
            tree_method="hist",
            enable_categorical=True,
            random_state=42 + i * 2 + rep,
            n_jobs=N_JOBS,
            **cfg,
        )
        m.fit(Xfull, yfull)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
