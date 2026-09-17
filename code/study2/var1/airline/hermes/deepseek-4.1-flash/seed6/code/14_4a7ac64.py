"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# busiest origins by training-slice volume (fitted on train only); used by the origin x hour cross
TOP_ORIGINS = set(train["Origin"].value_counts().head(30).index)

# --- features -----------------------------------------------------------------
# Month / DayofMonth are dropped: they encode year-specific seasonality that does not transfer from the
# 2005 training slice to later rows and measurably hurt generalization (-0.004 AUC when kept).
FEATURE_INPUTS = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Distance", "DepTime"]


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise feature engineering only: no statistics fitted on any dataset."""
    X = pd.DataFrame(index=df.index)
    for c in FEATURE_INPUTS:
        if c != "DepTime" and c != "Distance":
            X[c] = df[c]
    # scheduled departure is hhmm with hours past 24 for post-midnight red-eyes
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = dep // 100
    tod = hour * 60 + (dep % 100).clip(0, 59)
    X["dep_hour"] = hour
    X["dep_minute"] = (dep % 100).clip(0, 59)
    X["dep_tod"] = tod
    X["Distance"] = df["Distance"]
    # carrier x hour-of-day cross: carriers have distinct delay profiles through the day that transfer
    # across years (+0.016 AUC on its own, by far the strongest feature block so far)
    X["carrier_hour"] = df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
    # distance bucket x hour cross: long-haul vs short-haul flights have different delay profiles by
    # time of day (250-mile buckets worked best of the widths tried)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["dist_bin_hour"] = (dist // 250).astype(str) + "_" + hour.astype(str)
    # top-30 origin x hour: full 282-airport crosses overfit, but the busiest airports have reliable
    # hour-of-day delay profiles (rest folded into an "other" bucket)
    origin_top = df["Origin"].where(df["Origin"].isin(TOP_ORIGINS), "OTHER")
    X["origin_top_hour"] = origin_top.astype(str) + "_" + hour.astype(str)
    # 15-minute time-of-day bucket as its own categorical: lets trees treat the daily delay ramp
    # non-monotonically instead of bisecting the numeric time axis
    X["tod15"] = (tod // 15).astype(str)
    return X


X_train = raw_features(train)
cat_cols = [c for c in X_train.columns if pd.api.types.is_object_dtype(X_train[c])]
cat_levels = {c: pd.Index(sorted(X_train[c].dropna().unique())) for c in cat_cols}
feature_cols = list(X_train.columns)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = raw_features(df)[feature_cols]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    tree_method="hist",
    sampling_method="gradient_based",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
N_MODELS = 10  # depth-diverse bag: cross-depth averaging beat any single depth
MODEL_SPECS = [
    (depth, 5)
    for depth in (4, 4, 6, 6, 8, 8, 10, 10, 12, 12)
]

t0 = time.time()
X_tr, y_tr = prepare(train), to_y(train)
models = []
for i, (depth, mcw) in enumerate(MODEL_SPECS[:N_MODELS]):
    m = xgb.XGBClassifier(
        **{**PARAMS, "max_depth": depth, "min_child_weight": mcw, "random_state": SEED + 7 * i}
    )
    m.fit(X_tr, y_tr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
