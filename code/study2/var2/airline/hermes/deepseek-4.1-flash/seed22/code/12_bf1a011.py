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


def _hour(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") // 100).clip(lower=0, upper=24)


def _cnum(series: pd.Series) -> pd.Series:
    """Month/DayofMonth/DayOfWeek arrive as 'c-<n>' strings; tolerate plain integers too."""
    s = series.astype(str).str.replace("c-", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


# carrier x hour-of-day: whether a carrier is punctual depends strongly on the time of day
def _carrier_hour(df: pd.DataFrame) -> pd.Series:
    return df["UniqueCarrier"].astype(str) + "_" + _hour(df["DepTime"]).astype(int).astype(str)


CARRIER_HOUR_LEVELS = pd.Index(sorted(_carrier_hour(train).unique()))

# origin-hour traffic volume (how busy an airport is at that hour) - airport congestion proxy
_ORG_HOUR_TRAIN = train["Origin"].astype(str) + "_" + _hour(train["DepTime"]).astype(int).astype(str)
ORG_HOUR_COUNTS = _ORG_HOUR_TRAIN.value_counts()


def _org_hour(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + _hour(df["DepTime"]).astype(int).astype(str)


ORIGIN_MEAN_DIST = train.groupby("Origin")["Distance"].mean()
DEST_MEAN_DIST = train.groupby("Dest")["Distance"].mean()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # scheduled departure clock features: delay risk rises steeply through the day
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = _hour(df["DepTime"])
    X["dep_hour"] = hour
    X["dep_minofday"] = hour * 60 + dep % 100
    X["carrier_hour"] = pd.Categorical(_carrier_hour(df), categories=CARRIER_HOUR_LEVELS)
    # holiday periods and extreme parts of the day: delay probability spikes there
    mon = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    X["is_dec_holiday"] = ((mon == 12) & (dom >= 18)).astype(int)
    X["is_thanksgiving"] = ((mon == 11) & (dom >= 22)).astype(int)
    X["is_jul4"] = ((mon == 7) & (dom <= 7)).astype(int)
    X["is_newyear"] = (((mon == 12) & (dom >= 28)) | ((mon == 1) & (dom <= 5))).astype(int)
    X["is_late_hour"] = (hour >= 20).astype(int)
    X["is_early_hour"] = (hour <= 6).astype(int)
    X["org_hour_count"] = np.log1p(_org_hour(df).map(ORG_HOUR_COUNTS).fillna(0.0).to_numpy())
    # long-haul flights accumulate delay differently as the day progresses
    X["dist_x_hour"] = pd.to_numeric(df["Distance"], errors="coerce") / 1000.0 * hour
    # airport character: typical stage length of flights served there
    X["origin_mean_dist"] = df["Origin"].map(ORIGIN_MEAN_DIST)
    X["dest_mean_dist"] = df["Dest"].map(DEST_MEAN_DIST)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=600,
    max_depth=28,
    learning_rate=0.02,
    min_child_weight=2,
    reg_lambda=10.0,
    colsample_bytree=0.3,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 2024]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **PARAMS)
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
