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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# string columns are used as categoricals, except the raw 300-level airport codes: partition splits on those
# overfit one year's airport quirks, their information is carried smoothly by the traffic features below.
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000 and c not in ("Origin", "Dest", "DayofMonth", "Month")]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# raw columns the derived traffic features are keyed on (kept even when not model features themselves)
KEY_COLS = ["Origin", "Dest", "UniqueCarrier", "Month", "DayOfWeek", "DayofMonth"]


def _hour_of(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype("int64")
    return (t // 100).clip(0, 23)


def _key(df: pd.DataFrame, col: str, suffix) -> pd.Series:
    return df[col].astype(str) + "|" + suffix


def _cn(s: pd.Series) -> pd.Series:
    """'c-4' -> 4: the ordinal carried by these coded string columns."""
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


# --- train-only statistics (recomputed by prepare() on any incoming frame) -----
_hr = _hour_of(train).astype(str)
# hourly traffic volume of each airport / carrier, counted on the training year only
hour_counts = {c: _key(train, c, _hr).value_counts().to_dict() for c in ("Origin", "Dest", "UniqueCarrier")}

# US holiday travel peaks (date-anchored, no year needed)
_ANCHOR_DOY = np.array([(m - 1) * 30.4 + d for m, d in [(1, 1), (5, 30), (7, 4), (9, 4), (11, 24), (12, 25)]])

# (base column, which hourly histogram to read, feature prefix): the airport/carrier "wave" structure.
# A flight inherits the traffic of the airport it leaves from, the traffic arriving there (late aircraft),
# the traffic at its destination, and its own carrier's schedule wave.
PROFILE_SIDES = [
    ("Origin", "Origin", "orig_dep"),       # departures out of the origin airport
    ("Origin", "Dest", "orig_inb"),         # arrivals into the origin airport
    ("Dest", "Dest", "dest_dep"),           # arrivals into the destination airport
    ("UniqueCarrier", "UniqueCarrier", "car_dep"),  # the carrier's own schedule wave
]
LAGS = range(1, 13)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].drop(columns=["DepTime"], errors="ignore").copy()
    hour = _hour_of(df)
    X["dep_hour"] = hour
    X["dep_min"] = (pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype("int64") % 100).clip(0, 59)
    X["dep_tod"] = X["dep_hour"] * 60 + X["dep_min"]
    hs = hour.astype(str)

    for c in ("Origin", "Dest", "UniqueCarrier"):
        X[f"{c}_hour_count"] = _key(df, c, hs).map(hour_counts[c]).fillna(0).astype("float32")
    # aircraft arriving at the origin during this hour: late inbounds delay this departure
    inbound = _key(df, "Origin", hs).map(hour_counts["Dest"]).fillna(0).astype("float32")
    X["origin_inbound_hour_count"] = inbound
    X["origin_movements_hour"] = X["Origin_hour_count"] + inbound

    for base_col, side, tag in PROFILE_SIDES:
        for lag in LAGS:
            hl = (hour - lag).clip(0, 23).astype(str)
            X[f"{tag}_lag{lag}"] = _key(df, base_col, hl).map(hour_counts[side]).fillna(0).astype("float32")
        for lead in LAGS:
            hl = (hour + lead).clip(0, 23).astype(str)
            X[f"{tag}_lead{lead}"] = _key(df, base_col, hl).map(hour_counts[side]).fillna(0).astype("float32")

    season = _cn(df["Month"])
    X["month_i"] = season.astype("int8")
    X["month_sin"] = np.sin(2 * np.pi * season / 12).astype("float32")
    X["month_cos"] = np.cos(2 * np.pi * season / 12).astype("float32")
    mdoy = (season - 1) * 30.4 + _cn(df["DayofMonth"])
    dist = np.min(np.abs(mdoy.to_numpy()[:, None] - _ANCHOR_DOY[None, :]), axis=1)
    X["days_to_holiday"] = np.minimum(dist, 30).astype("float32")
    X["is_holiday_week"] = (dist <= 3).astype("int8")

    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# bagged XGBoost: averaging models that actually differ (row/column subsampling, depth, seed) is the
# variance reduction that carries over to the hidden holdout
ENSEMBLE = [
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=1.0, colsample_bytree=0.7, random_state=42),
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.7, colsample_bytree=1.0, random_state=7),
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=2024),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9, random_state=1337),
    dict(n_estimators=500, max_depth=7, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=99),
    dict(n_estimators=800, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7, random_state=2025),
]
COMMON = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
models = []
for p in ENSEMBLE:
    m = xgb.XGBClassifier(**COMMON, **p)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
