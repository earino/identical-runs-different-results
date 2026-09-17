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
from sklearn.model_selection import KFold

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
# ablation: high-cardinality raw categoricals are represented by TE/count features
_DROP_RAW = {"Origin", "Dest", "UniqueCarrier"}
cat_cols = [c for c in cat_cols if c not in _DROP_RAW]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# frequency (count) encodings fitted on training data only
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_hour_tr = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 23)
_freq_maps = {
    "freq_route": _route_tr.value_counts(),
    "freq_origin": train["Origin"].astype(str).value_counts(),
    "freq_dest": train["Dest"].astype(str).value_counts(),
    "freq_carrier": train["UniqueCarrier"].astype(str).value_counts(),
    # congestion proxies: how many flights leave/arrive at this airport in this hour / season
    "freq_origin_hour": (train["Origin"].astype(str) + "_" + _hour_tr.astype(str)).value_counts(),
    "freq_dest_hour": (train["Dest"].astype(str) + "_" + _hour_tr.astype(str)).value_counts(),
    "freq_dow_hour": (train["DayOfWeek"].astype(str) + "_" + _hour_tr.astype(str)).value_counts(),
    "freq_origin_month": (train["Origin"].astype(str) + "_" + train["Month"].astype(str)).value_counts(),
    "freq_carrier_origin": (train["UniqueCarrier"].astype(str) + "_" + train["Origin"].astype(str)).value_counts(),
    "freq_dest_month": (train["Dest"].astype(str) + "_" + train["Month"].astype(str)).value_counts(),
    "freq_route_hour": (train["Origin"].astype(str) + "_" + train["Dest"].astype(str) + "_" + _hour_tr.astype(str)).value_counts(),
    "freq_carrier_route": (train["UniqueCarrier"].astype(str) + "_" + train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts(),
    "freq_carrier_hour": (train["UniqueCarrier"].astype(str) + "_" + _hour_tr.astype(str)).value_counts(),
    # structural: hub size / route competition (input-only, robust)
    "n_carriers_route": train.assign(_r=_route_tr, _c=train["UniqueCarrier"].astype(str)).groupby("_r")["_c"].nunique(),
    "n_dest_origin": train.assign(_o=train["Origin"].astype(str)).groupby("_o")["Dest"].nunique(),
    "n_orig_dest": train.assign(_d=train["Dest"].astype(str)).groupby("_d")["Origin"].nunique(),
}


# --- target (delay-rate) encodings, fitted on training data only ----------------
def _te_keys(df: pd.DataFrame) -> pd.DataFrame:
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23).astype("Int64").astype(str)
    half = (((dep // 100).clip(0, 23)) * 2 + ((dep % 100) // 30)).astype("Int64").astype(str)
    quarter = (((dep // 100).clip(0, 23)) * 4 + ((dep % 100) // 15)).astype("Int64").astype(str)
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    c = df["UniqueCarrier"].astype(str)
    return pd.DataFrame(
        {
            "te_origin": o.values,
            "te_dest": d.values,
            "te_carrier": c.values,
            "te_route": (o + "_" + d).values,
            "te_origin_hour": (o + "_" + hour).values,
            "te_dest_hour": (d + "_" + hour).values,
            "te_carrier_hour": (c + "_" + hour).values,
            "te_origin_month": (o + "_" + df["Month"].astype(str)).values,
            "te_dest_month": (d + "_" + df["Month"].astype(str)).values,
            "te_route_hour": (o + "_" + d + "_" + hour).values,
            "te_carrier_route": (c + "_" + o + "_" + d).values,
            "te_origin_hour_month": (o + "_" + hour + "_" + df["Month"].astype(str)).values,
            "te_carrier_route_hour": (c + "_" + o + "_" + d + "_" + hour).values,
            "te_dest_hour_month": (d + "_" + hour + "_" + df["Month"].astype(str)).values,
            "te_origin_dow_hour": (o + "_" + df["DayOfWeek"].astype(str) + "_" + hour).values,
            "te_dest_dow_hour": (d + "_" + df["DayOfWeek"].astype(str) + "_" + hour).values,
            "te_carrier_origin_hour": (c + "_" + o + "_" + hour).values,
            "te_carrier_dest_hour": (c + "_" + d + "_" + hour).values,
            "te_carrier_hour_month": (c + "_" + hour + "_" + df["Month"].astype(str)).values,
            "te_route_dow_hour": (o + "_" + d + "_" + df["DayOfWeek"].astype(str) + "_" + hour).values,
            "te_route_half": (o + "_" + d + "_" + half).values,
            "te_origin_half": (o + "_" + half).values,
            "te_dest_half": (d + "_" + half).values,
            "te_carrier_half": (c + "_" + half).values,
            "te_route_quarter": (o + "_" + d + "_" + quarter).values,
            "te_origin_quarter": (o + "_" + quarter).values,
            "te_dest_quarter": (d + "_" + quarter).values,
            "te_carrier_quarter": (c + "_" + quarter).values,
        }
    )


SMOOTH = 30.0
_y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(_y.mean())
_kt = _te_keys(train)
TE_COLS = list(_kt.columns)


def _fit_map(keys: np.ndarray, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)


te_maps = {c: _fit_map(_kt[c].to_numpy(), _y) for c in TE_COLS}
# support (count) of the finer interaction keys, so the model knows how reliable each TE is
_SUPP = [
    "te_origin_hour_month",
    "te_dest_hour_month",
    "te_carrier_origin_hour",
    "te_carrier_dest_hour",
    "te_carrier_hour_month",
    "te_route_dow_hour",
    "te_origin_dow_hour",
    "te_dest_dow_hour",
    "te_route_half",
    "te_origin_half",
    "te_dest_half",
    "te_carrier_half",
    "te_route_quarter",
    "te_origin_quarter",
    "te_dest_quarter",
    "te_carrier_quarter",
]
te_supp = {c: _kt[c].value_counts() for c in _SUPP}
# out-of-fold encodings for the training rows (avoids target leakage during fit)
_oof = np.full((len(train), len(TE_COLS)), PRIOR, dtype=float)
for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(len(train))):
    for _j, _c in enumerate(TE_COLS):
        _m = _fit_map(_kt[_c].to_numpy()[_tr], _y[_tr])
        _oof[_va, _j] = pd.Series(_kt[_c].to_numpy()[_va]).map(_m).fillna(PRIOR).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # scheduled departure time: hhmm integer -> continuous time-of-day
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_tod"] = ((dep // 100).clip(0, 23) + (dep % 100) / 60.0).to_numpy()
    # count encodings
    route = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    hour = (dep // 100).clip(0, 23)
    X["freq_route"] = pd.Series(route).map(_freq_maps["freq_route"]).fillna(0).to_numpy()
    X["freq_origin"] = df["Origin"].astype(str).map(_freq_maps["freq_origin"]).fillna(0).to_numpy()
    X["freq_dest"] = df["Dest"].astype(str).map(_freq_maps["freq_dest"]).fillna(0).to_numpy()
    X["freq_carrier"] = df["UniqueCarrier"].astype(str).map(_freq_maps["freq_carrier"]).fillna(0).to_numpy()
    X["freq_origin_hour"] = (
        df["Origin"].astype(str) + "_" + hour.astype("Int64").astype(str)
    ).map(_freq_maps["freq_origin_hour"]).fillna(0).to_numpy()
    X["freq_dest_hour"] = (
        df["Dest"].astype(str) + "_" + hour.astype("Int64").astype(str)
    ).map(_freq_maps["freq_dest_hour"]).fillna(0).to_numpy()
    X["freq_dow_hour"] = (
        df["DayOfWeek"].astype(str) + "_" + hour.astype("Int64").astype(str)
    ).map(_freq_maps["freq_dow_hour"]).fillna(0).to_numpy()
    X["freq_origin_month"] = (
        df["Origin"].astype(str) + "_" + df["Month"].astype(str)
    ).map(_freq_maps["freq_origin_month"]).fillna(0).to_numpy()
    X["freq_carrier_origin"] = (
        df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    ).map(_freq_maps["freq_carrier_origin"]).fillna(0).to_numpy()
    X["freq_dest_month"] = (
        df["Dest"].astype(str) + "_" + df["Month"].astype(str)
    ).map(_freq_maps["freq_dest_month"]).fillna(0).to_numpy()
    hour_s = hour.astype("Int64").astype(str)
    X["freq_route_hour"] = (
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + hour_s
    ).map(_freq_maps["freq_route_hour"]).fillna(0).to_numpy()
    X["freq_carrier_route"] = (
        df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    ).map(_freq_maps["freq_carrier_route"]).fillna(0).to_numpy()
    X["freq_carrier_hour"] = (
        df["UniqueCarrier"].astype(str) + "_" + hour_s
    ).map(_freq_maps["freq_carrier_hour"]).fillna(0).to_numpy()
    X["n_carriers_route"] = pd.Series(route).map(_freq_maps["n_carriers_route"]).fillna(0).to_numpy()
    X["n_dest_origin"] = df["Origin"].astype(str).map(_freq_maps["n_dest_origin"]).fillna(0).to_numpy()
    X["n_orig_dest"] = df["Dest"].astype(str).map(_freq_maps["n_orig_dest"]).fillna(0).to_numpy()
    # shares: how concentrated this airport/route is at this hour
    X["share_origin_hour"] = (X["freq_origin_hour"] / X["freq_origin"].clip(lower=1)).to_numpy()
    X["share_route"] = (X["freq_route"] / X["freq_origin"].clip(lower=1)).to_numpy()
    # target (delay-rate) encodings
    keys = _te_keys(df)
    for c in TE_COLS:
        X[c] = keys[c].map(te_maps[c]).fillna(PRIOR).to_numpy()
    for c in _SUPP:
        X["n_" + c] = keys[c].map(te_supp[c]).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=4, colsample_bytree=0.8, subsample=0.8),
    dict(max_depth=5, colsample_bytree=0.8, subsample=0.8),
    dict(max_depth=3, colsample_bytree=0.8, subsample=0.8),
    dict(max_depth=4, colsample_bytree=0.6, subsample=0.9),
    dict(max_depth=5, colsample_bytree=0.6, subsample=0.7),
    dict(max_depth=4, colsample_bytree=1.0, subsample=0.7),
    dict(max_depth=5, colsample_bytree=0.8, subsample=0.9),
    dict(max_depth=3, colsample_bytree=0.6, subsample=0.9),
    dict(max_depth=4, colsample_bytree=0.7, subsample=0.75),
    dict(max_depth=5, colsample_bytree=0.7, subsample=0.85),
    dict(max_depth=4, colsample_bytree=0.9, subsample=0.6),
    dict(max_depth=5, colsample_bytree=0.5, subsample=0.8),
]
models = [
    xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.04,
        min_child_weight=50,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
        **cfg,
    )
    for i, cfg in enumerate(CONFIGS)
]

t0 = time.time()
X_train = prepare(train)
X_train[TE_COLS] = _oof
y_train = to_y(train)
for m in models:
    m.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
