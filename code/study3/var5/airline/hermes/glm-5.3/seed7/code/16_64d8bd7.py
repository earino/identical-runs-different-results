"""XGBoost airline-delay classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
hour_levels = [str(i) for i in range(25)]
# count/frequency encodings (target-agnostic, fit on train only)
route_counts = (train["Origin"].astype(str) + "-" + train["Dest"].astype(str)).value_counts().to_dict()
origin_counts = train["Origin"].value_counts().to_dict()
dest_counts = train["Dest"].value_counts().to_dict()
# schedule-pressure counts: how many train flights share this (origin, 15-min block), etc.
_mins_tr = (train["DepTime"].astype(int) // 100) * 60 + train["DepTime"].astype(int) % 100
bucket15_levels = pd.Index(sorted(((_mins_tr // 15) * 15).astype(str).unique()))
orig_block_counts = pd.DataFrame({"o": train["Origin"], "b": (_mins_tr // 15) * 15}).value_counts().to_dict()
orig_hour_counts = pd.DataFrame({"o": train["Origin"], "h": _mins_tr // 60}).value_counts().to_dict()
block_counts = ((_mins_tr // 15) * 15).value_counts().to_dict()
dest_block_counts = pd.DataFrame({"d": train["Dest"], "b": (_mins_tr // 15) * 15}).value_counts().to_dict()
carrier_hour_counts = pd.DataFrame({"c": train["UniqueCarrier"], "h": _mins_tr // 60}).value_counts().to_dict()
route_hour_counts = pd.DataFrame(
    {"r": train["Origin"].astype(str) + "-" + train["Dest"].astype(str), "h": _mins_tr // 60}
).value_counts().to_dict()


def prepare(df: pd.DataFrame, variant: int = 0) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepTime", "Distance"]].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    # cyclic day-of-year (Month/DayofMonth are c-<n> strings)
    month = df["Month"].str.slice(2).astype(float)
    day = df["DayofMonth"].str.slice(2).astype(float)
    doy = (month - 1.0) * 30.4 + day
    X["SeasonSin"] = np.sin(2 * np.pi * doy / 365.0)
    X["SeasonCos"] = np.cos(2 * np.pi * doy / 365.0)
    # hour-of-day as categorical
    hour = np.floor(df["DepTime"].astype(float) / 100.0).clip(0, 24).astype(int).astype(str)
    X["DepHour"] = pd.Categorical(hour, categories=hour_levels)
    # log flight counts per route/origin/dest (train-fit)
    route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    X["RouteFreq"] = np.log1p(route.map(route_counts).fillna(0.0))
    X["OriginFreq"] = np.log1p(df["Origin"].map(origin_counts).fillna(0.0))
    X["DestFreq"] = np.log1p(df["Dest"].map(dest_counts).fillna(0.0))
    # true minutes since midnight (DepTime is hhmm: 1159 and 1200 are 1 min apart, not 41)
    hhmm = df["DepTime"].astype(int)
    mins = (hhmm // 100) * 60 + hhmm % 100
    X["DepMinutes"] = mins.astype(float)
    b15 = (mins // 15) * 15
    X["DepBucket15"] = pd.Categorical(b15.astype(str), categories=bucket15_levels)
    # schedule pressure at this departure time/place (train-fit counts)
    X["OrigBlockCount"] = [orig_block_counts.get((o, b), 0) for o, b in zip(df["Origin"], b15)]
    X["OrigHourCount"] = [orig_hour_counts.get((o, h), 0) for o, h in zip(df["Origin"], mins // 60)]
    X["BlockCount"] = [block_counts.get(b, 0) for b in b15]
    if variant == 1:
        X["DestBlockCount"] = [dest_block_counts.get((d, b), 0) for d, b in zip(df["Dest"], b15)]
        X["CarrierHourCount"] = [carrier_hour_counts.get((c, h), 0) for c, h in zip(df["UniqueCarrier"], mins // 60)]
        X["RouteHourCount"] = [route_hour_counts.get((r, h), 0) for r, h in zip(route, mins // 60)]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# heterogeneous ensemble: two feature variants x feature-dropped subsets x seeds
ENSEMBLE = [
    (0, 42, []),
    (0, 7, ["Distance"]),
    (0, 2024, ["SeasonSin", "SeasonCos"]),
    (1, 13, ["Dest"]),
    (1, 99, []),
]
PARAMS = dict(
    n_estimators=600,
    max_depth=10,
    learning_rate=0.025,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
)
# recency weighting: 2005 rows later in the year are closer in time to the 2006 eval,
# so up-weight them mildly (linear ramp from 0.5 in January to 1.0 in December).
_month_num = train["Month"].str.slice(2).astype(float).to_numpy()
SAMPLE_WEIGHTS = 0.5 + 0.5 * (_month_num / 12.0)

t0 = time.time()
y_train = to_y(train)
models = []
for variant, seed, drop in ENSEMBLE:
    Xv = prepare(train, variant)
    feats = [c for c in Xv.columns if c not in drop]
    m = xgb.XGBClassifier(random_state=seed, n_jobs=N_JOBS, **PARAMS)
    m.fit(Xv[feats], y_train, sample_weight=SAMPLE_WEIGHTS)
    models.append((variant, drop, m))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xs = {v: prepare(df, v) for v in {var for var, _, _ in ENSEMBLE}}
    return np.mean([m.predict_proba(Xs[v][[c for c in Xs[v].columns if c not in drop]])[:, 1]
                    for v, drop, m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
