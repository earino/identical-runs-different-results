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
# Day-of-month has no plausible mechanism for departure delays; it is 31 levels of pure noise to fit.
feature_cols = [c for c in feature_cols if c != "DayofMonth"]
cat_cols = [c for c in cat_cols if c != "DayofMonth"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# Traffic-volume features: how busy an airport / route / airport-hour is. Counts are fit on TRAINING data
# only and are far more stable across the year boundary than any delay-rate statistic.
_t_hour = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 23)
_t_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
VOL_MAPS = {
    "OriginVol": train["Origin"].value_counts().to_dict(),
    "DestVol": train["Dest"].value_counts().to_dict(),
    "RouteVol": _t_route.value_counts().to_dict(),
    "OriginHourVol": (train["Origin"].astype(str) + "_" + _t_hour.astype(int).astype(str)).value_counts().to_dict(),
    "CarrierVol": train["UniqueCarrier"].value_counts().to_dict(),
    "CarrierRouteVol": (train["UniqueCarrier"].astype(str) + "_" + _t_route).value_counts().to_dict(),
    "DestHourVol": (train["Dest"].astype(str) + "_" + _t_hour.astype(int).astype(str)).value_counts().to_dict(),
    "HourVol": _t_hour.astype(int).value_counts().to_dict(),
    "OriginCarrierVol": (train["Origin"].astype(str) + "_" + train["UniqueCarrier"].astype(str)).value_counts().to_dict(),
}


# Peak-shape ratios: how peaky this hour is for the airport / for the system as a whole.
_oh_counts = (train["Origin"].astype(str) + "_" + _t_hour.astype(int).astype(str)).value_counts()
_oh_max = _oh_counts.groupby([k.split("_")[0] for k in _oh_counts.index]).max().to_dict()
HV_MAX = float(max(VOL_MAPS["HourVol"].values()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # Coarse time-of-day features: they transfer across the 2005->2006 shift far better than the raw
    # hhmm integer, which lets the trees carve year-specific minute-level pockets.
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    X["Hour"] = hour
    X["TOD"] = hour * 60 + (dep % 100).clip(0, 59)
    X["IsWeekend"] = (pd.to_numeric(X["DayOfWeek"].astype(str).str.replace("c-", "", regex=False),
                                    errors="coerce") >= 6).astype(int)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["OriginVol"] = df["Origin"].map(VOL_MAPS["OriginVol"]).fillna(0)
    X["DestVol"] = df["Dest"].map(VOL_MAPS["DestVol"]).fillna(0)
    X["RouteVol"] = route.map(VOL_MAPS["RouteVol"]).fillna(0)
    X["OriginHourVol"] = (df["Origin"].astype(str) + "_" + hour.astype(int).astype(str)).map(
        VOL_MAPS["OriginHourVol"]).fillna(0)
    X["CarrierVol"] = df["UniqueCarrier"].map(VOL_MAPS["CarrierVol"]).fillna(0)
    # Scale-free traffic-structure shares: they describe how a flight sits inside the network
    # (hub-hour load, route load, carrier's grip on the route) and carry no year-specific rates.
    dh_key = df["Dest"].astype(str) + "_" + hour.astype(int).astype(str)
    X["OriginHourShare"] = X["OriginHourVol"] / X["OriginVol"].clip(lower=1)
    X["RouteShare"] = X["RouteVol"] / X["OriginVol"].clip(lower=1)
    X["CarrierRouteShare"] = (df["UniqueCarrier"].astype(str) + "_" + route).map(
        VOL_MAPS["CarrierRouteVol"]).fillna(0) / X["RouteVol"].clip(lower=1)
    X["DestHourVol"] = dh_key.map(VOL_MAPS["DestHourVol"]).fillna(0)
    X["HourVol"] = hour.astype(int).map(VOL_MAPS["HourVol"]).fillna(0)
    X["OriginHourPeak"] = X["OriginHourVol"] / df["Origin"].map(_oh_max).fillna(1).clip(lower=1)
    X["HourVolShare"] = X["HourVol"] / HV_MAX
    X["DestHourShare"] = X["DestHourVol"] / X["DestVol"].clip(lower=1)
    X["OriginCarrierShare"] = (df["Origin"].astype(str) + "_" + df["UniqueCarrier"].astype(str)).map(
        VOL_MAPS["OriginCarrierVol"]).fillna(0) / X["OriginVol"].clip(lower=1)
    # Cyclic encodings give the trees smooth, year-independent handles on time-of-day and season.
    month = pd.to_numeric(X["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    X["HourSin"] = np.sin(2 * np.pi * hour / 24.0)
    X["HourCos"] = np.cos(2 * np.pi * hour / 24.0)
    X["MonthSin"] = np.sin(2 * np.pi * month / 12.0)
    X["MonthCos"] = np.cos(2 * np.pi * month / 12.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# A shallow, well-regularized booster is the best regime here: deeper/longer fits memorize the 2005
# slice and transfer worse to the later-period eval rows. Averaging diverse seeds/depths cuts that variance.
BASE_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.1,
    reg_lambda=3.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
# (max_depth, seed offset) per ensemble member: a spread of depths hedges against any single depth
# being the one that happens to overfit this particular train slice.
# (max_depth, subsample, colsample_bytree, seed offset) per member.
MEMBERS = [(d, 0.65 + 0.05 * (i % 4), 0.5 + 0.05 * (i % 8), i)
           for i, d in enumerate([3, 4, 5, 6] * 15)]
X_train, y_train = prepare(train), to_y(train)

t0 = time.time()
models = []
for depth, sub, cols, soff in MEMBERS:
    m = xgb.XGBClassifier(max_depth=depth, subsample=sub, colsample_bytree=cols,
                          random_state=SEED + 101 * soff + depth, **BASE_PARAMS)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
