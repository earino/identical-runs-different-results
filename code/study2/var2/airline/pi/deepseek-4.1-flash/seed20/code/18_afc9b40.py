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
tr_tod = tr_hour * 60 + (train["DepTime"].fillna(0).astype(int) % 100).clip(0, 59)
tr_bin = (tr_tod // 15).astype(str)
cnt_origin_bin = (train["Origin"].astype(str) + "_" + tr_bin).value_counts()
cnt_dest_bin = (train["Dest"].astype(str) + "_" + tr_bin).value_counts()
cnt_route_bin = (_tr_route + "_" + tr_bin).value_counts()
cnt_route_dow = (_tr_route + "_" + tr_dow).value_counts()
cnt_origin_day = (
    train["Origin"].astype(str)
    + "_"
    + train["Month"].astype(str)
    + "_"
    + train["DayofMonth"].astype(str)
).value_counts()
tr_bin2 = (tr_tod // 120).astype(str)
cnt_origin_2h = (train["Origin"].astype(str) + "_" + tr_bin2).value_counts()
cnt_dest_2h = (train["Dest"].astype(str) + "_" + tr_bin2).value_counts()
cnt_carrier_bin = (train["UniqueCarrier"].astype(str) + "_" + tr_bin).value_counts()
cnt_carrier_route = (
    train["UniqueCarrier"].astype(str) + "_" + _tr_route
).value_counts()
tr_month = train["Month"].astype(str)
cnt_origin_month = (train["Origin"].astype(str) + "_" + tr_month).value_counts()
cnt_dest_month = (train["Dest"].astype(str) + "_" + tr_month).value_counts()
cnt_route_2h = (_tr_route + "_" + tr_bin2).value_counts()


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
    bin_s = (X["tod"] // 15).astype(str)
    X["origin_bin_cnt"] = (df["Origin"].astype(str) + "_" + bin_s).map(cnt_origin_bin).fillna(0).astype(float)
    X["dest_bin_cnt"] = (df["Dest"].astype(str) + "_" + bin_s).map(cnt_dest_bin).fillna(0).astype(float)
    X["route_bin_cnt"] = (
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + bin_s
    ).map(cnt_route_bin).fillna(0).astype(float)
    X["route_dow_cnt"] = (
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + dow_s
    ).map(cnt_route_dow).fillna(0).astype(float)
    X["origin_day_cnt"] = (
        df["Origin"].astype(str) + "_" + df["Month"].astype(str) + "_" + df["DayofMonth"].astype(str)
    ).map(cnt_origin_day).fillna(0).astype(float)
    X["origin_bin_frac"] = X["origin_bin_cnt"] / df["Origin"].astype(str).map(cnt_origin).fillna(1).astype(float)
    X["dest_bin_frac"] = X["dest_bin_cnt"] / df["Dest"].astype(str).map(cnt_dest).fillna(1).astype(float)
    X["route_bin_frac"] = X["route_bin_cnt"] / X["route_cnt"].replace(0, 1)
    bin2_s = (X["tod"] // 120).astype(str)
    X["origin_2h_cnt"] = (df["Origin"].astype(str) + "_" + bin2_s).map(cnt_origin_2h).fillna(0).astype(float)
    X["dest_2h_cnt"] = (df["Dest"].astype(str) + "_" + bin2_s).map(cnt_dest_2h).fillna(0).astype(float)
    X["carrier_bin_cnt"] = (
        df["UniqueCarrier"].astype(str) + "_" + bin_s
    ).map(cnt_carrier_bin).fillna(0).astype(float)
    X["carrier_bin_frac"] = X["carrier_bin_cnt"] / df["UniqueCarrier"].astype(str).map(cnt_carrier).fillna(1).astype(float)
    route_s = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["carrier_route_cnt"] = (
        df["UniqueCarrier"].astype(str) + "_" + route_s
    ).map(cnt_carrier_route).fillna(0).astype(float)
    month_s = df["Month"].astype(str)
    X["origin_month_cnt"] = (df["Origin"].astype(str) + "_" + month_s).map(cnt_origin_month).fillna(0).astype(float)
    X["dest_month_cnt"] = (df["Dest"].astype(str) + "_" + month_s).map(cnt_dest_month).fillna(0).astype(float)
    X["route_2h_cnt"] = (route_s + "_" + bin2_s).map(cnt_route_2h).fillna(0).astype(float)
    prev_bin_s = ((X["tod"] // 15) - 1).clip(0, 95).astype(str)
    next_bin_s = ((X["tod"] // 15) + 1).clip(0, 95).astype(str)
    X["origin_prev_bin_cnt"] = (df["Origin"].astype(str) + "_" + prev_bin_s).map(cnt_origin_bin).fillna(0).astype(float)
    X["origin_next_bin_cnt"] = (df["Origin"].astype(str) + "_" + next_bin_s).map(cnt_origin_bin).fillna(0).astype(float)
    X["dest_prev_bin_cnt"] = (df["Dest"].astype(str) + "_" + prev_bin_s).map(cnt_dest_bin).fillna(0).astype(float)
    X["dest_next_bin_cnt"] = (df["Dest"].astype(str) + "_" + next_bin_s).map(cnt_dest_bin).fillna(0).astype(float)
    X["route_prev_bin_cnt"] = (route_s + "_" + prev_bin_s).map(cnt_route_bin).fillna(0).astype(float)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=9, colsample_bytree=0.5, min_child_weight=20, reg_lambda=5.0, n_estimators=700, learning_rate=0.02),
    dict(max_depth=10, colsample_bytree=0.6, min_child_weight=40, reg_lambda=10.0, n_estimators=600, learning_rate=0.02),
    dict(max_depth=11, colsample_bytree=0.6, min_child_weight=80, reg_lambda=20.0, n_estimators=500, learning_rate=0.02),
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
