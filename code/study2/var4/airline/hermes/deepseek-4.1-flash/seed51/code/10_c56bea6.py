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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- frequency / congestion encodings, fitted on the 2005 training data only ---
# Flight volumes per airport/route/hour are a stable structural property of the network, so they should
# transfer across years; no target information is involved.


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype("int64") % 2400
    hour = (dep // 100).astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    return pd.DataFrame({
        "route": origin + "_" + dest,
        "origin": origin,
        "dest": dest,
        "carrier": carrier,
        "origin_hour": origin + "_" + hour,
        "dest_hour": dest + "_" + hour,
        "carrier_hour": carrier + "_" + hour,
        "route_hour": origin + "_" + dest + "_" + hour,
        "carrier_origin": carrier + "_" + origin,
    })


CNT_MAP = {k: _keys(train)[k].value_counts() for k in _keys(train).columns}

# structural (target-free) descriptions of the route network, fitted on 2005 only
_keys_train = _keys(train)
_carrier_train = train["UniqueCarrier"].astype(str)
NCAR_ROUTE = _keys_train.assign(carrier=_carrier_train).groupby("route")["carrier"].nunique()
CARRIER_ROUTE_CNT = (_keys_train["route"] + "|" + _carrier_train).value_counts()
del _keys_train, _carrier_train


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype("int64") % 2400
    X["dep_hour"] = (dep // 100).astype("float64")
    X["dep_minute"] = (dep % 100).astype("float64")
    X["dep_frac"] = X["dep_hour"] + X["dep_minute"] / 60.0
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_frac"] / 24.0)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_frac"] / 24.0)
    X["next_day"] = ((pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) >= 2400)).astype("int8")
    # ordinal duplicates of the seasonal columns: trees split monotonically on these, which transfers
    # better across years than the unordered categorical versions alone
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["month_num"] = month.astype("float64")
    X["dom_num"] = dom.astype("float64")
    X["dow_num"] = dow.astype("float64")
    X["month_sin"] = np.sin(2 * np.pi * X["month_num"] / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * X["month_num"] / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow_num"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow_num"] / 7.0)
    X["is_weekend"] = (X["dow_num"] >= 6).astype("int8")
    X["day_of_year"] = (X["month_num"] - 1) * 30.44 + X["dom_num"]
    keys = _keys(df)
    for k in keys.columns:
        X[f"cnt_{k}"] = keys[k].map(CNT_MAP[k]).astype("float64").fillna(0.0)
    X["ncar_route"] = keys["route"].map(NCAR_ROUTE).astype("float64").fillna(1.0)
    X["carrier_route_cnt"] = (
        keys["route"] + "|" + df["UniqueCarrier"].astype(str)
    ).map(CARRIER_ROUTE_CNT).astype("float64").fillna(0.0)
    X["route_share"] = X["cnt_route"] / X["cnt_origin"].replace(0.0, np.nan)
    X["origin_hour_share"] = X["cnt_origin_hour"] / X["cnt_origin"].replace(0.0, np.nan)
    X["carrier_share"] = X["cnt_carrier_origin"] / X["cnt_origin"].replace(0.0, np.nan)
    X["route_dest_share"] = X["cnt_route"] / X["cnt_dest"].replace(0.0, np.nan)
    X["route_hour_share"] = X["cnt_route_hour"] / X["cnt_origin_hour"].replace(0.0, np.nan)
    X["carrier_route_share"] = X["carrier_route_cnt"] / X["cnt_route"].replace(0.0, np.nan)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# XGBoost has no seed-ensembling built in; averaging a few diverse boosters reduces variance, which is the
# dominant source of error for a 2005-trained model scored on 2006 rows.
BASE_PARAMS = dict(
    n_estimators=900,
    max_depth=6,
    learning_rate=0.025,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

MEMBER_OVERRIDES = [
    dict(random_state=SEED),
    dict(random_state=SEED + 1, subsample=0.7, colsample_bytree=0.6),
    dict(random_state=SEED + 2, max_depth=5, min_child_weight=10, n_estimators=1200),
    dict(random_state=SEED + 3, max_depth=8, min_child_weight=50, colsample_bytree=0.5),
    dict(random_state=SEED + 4, learning_rate=0.05, n_estimators=500, subsample=0.9),
]

X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
for ov in MEMBER_OVERRIDES:
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **ov})
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
