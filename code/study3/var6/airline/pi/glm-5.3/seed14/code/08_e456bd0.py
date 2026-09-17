"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design (all choices validated on the time-shifted eval year):
  - Time-of-day features dominate and are stable year-over-year (hhmm -> hour, minutes, minute-of-hour,
    cyclic sin/cos, plus an estimated arrival time from distance/450mph and its cyclic encoding).
  - Structural schedule counts (airport/carrier volume, origin/dest x hour congestion) transfer across years,
    but label-derived statistics (target encodings, rates) do NOT; route-level features are avoided entirely.
  - Small groups do NOT transfer 2005->2006: rare airports are pooled into NaN (missing) before training.
  - Once noisy rare levels are pooled, deep interaction trees become viable and transfer well: use a
    deep-emphasized ladder of XGBoost tree depths and average their probabilities (12 models, few rounds).
  - Oct-Dec 2005 rows are upweighted 2x: recency helps the following-year holdout.
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

# --- feature engineering ---------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}
# rare airports have noisy, non-transferable delay statistics across years: pool levels with < MIN_COUNT
# training rows into NaN (missing). Counts are computed on train only, never on the input dataframe.
MIN_COUNT = 300
cat_counts = {c: train[c].astype(str).value_counts() for c in ["Origin", "Dest", "UniqueCarrier"]}
_hr = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 24).astype(int)
cat_counts["oh"] = (train["Origin"].astype(str) + "@" + _hr.astype(str)).value_counts()
cat_counts["dh"] = (train["Dest"].astype(str) + "@" + _hr.astype(str)).value_counts()
cat_counts["h"] = _hr.astype(str).value_counts()
cat_counts["ch"] = (train["UniqueCarrier"].astype(str) + "@" + _hr.astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here; predict_proba() calls it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    # calendar: c-<n> strings -> ints
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = pd.to_numeric(df[c].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    # scheduled departure hhmm -> hour of day, minutes since midnight, minute of hour
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24).astype(int)
    X["DepHour"] = hour
    X["DepMinutes"] = hour * 60 + (dep % 100).clip(0, 59)
    X["MinuteOfHour"] = (dep % 100).clip(0, 59)
    frac = (X["DepMinutes"] % 1440) / 1440.0
    X["sin_time"] = np.sin(2 * np.pi * frac)
    X["cos_time"] = np.cos(2 * np.pi * frac)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    _hr = (dep // 100).clip(0, 24).astype(int).astype(str)
    X["carrier_cnt"] = df["UniqueCarrier"].astype(str).map(cat_counts["UniqueCarrier"]).fillna(0)
    X["origin_cnt"] = df["Origin"].astype(str).map(cat_counts["Origin"]).fillna(0)
    X["oh_cnt"] = (df["Origin"].astype(str) + "@" + _hr).map(cat_counts["oh"]).fillna(0)
    X["oh_share"] = X["oh_cnt"] / (X["origin_cnt"] + 1.0)
    X["h_cnt"] = _hr.map(cat_counts["h"]).fillna(0)
    X["dh_cnt"] = (df["Dest"].astype(str) + "@" + _hr).map(cat_counts["dh"]).fillna(0)
    X["dest_cnt"] = df["Dest"].astype(str).map(cat_counts["Dest"]).fillna(0)
    X["dh_share"] = X["dh_cnt"] / (X["dest_cnt"] + 1.0)
    X["ch_cnt"] = (df["UniqueCarrier"].astype(str) + "@" + _hr).map(cat_counts["ch"]).fillna(0)
    X["arr_minutes"] = X["DepMinutes"] + (dist / 450.0) * 60.0
    X["arr_hour"] = (X["arr_minutes"] // 60).clip(0, 40)
    afrac = (X["arr_minutes"] % 1440) / 1440.0
    X["arr_sin"] = np.sin(2 * np.pi * afrac)
    X["arr_cos"] = np.cos(2 * np.pi * afrac)
    X["arr_night"] = ((X["arr_minutes"] % 1440) >= 21 * 60).astype(int)
    for c in CAT_COLS:
        vals = df[c].astype(str)
        if c in cat_counts:  # pool rare Origin/Dest levels
            vals = vals.where(vals.map(cat_counts[c]).fillna(0) >= MIN_COUNT, other=np.nan)
        X[c] = pd.Categorical(vals, categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- training --------------------------------------------------------------------
y = to_y(train)
# recency weighting: the eval/holdout year follows late 2005, so upweight Oct-Dec 2005
month = pd.to_numeric(train["Month"].astype(str).str.replace("c-", "", regex=False)).to_numpy()
w = np.where(month >= 10, 2.0, 1.0)

# deep-emphasized depth ladder (12 members, few boosting rounds each); averaging probabilities
# over the ladder beats any single depth and is stable across seed sets
BAG_DEPTHS = [4, 8, 12, 16, 20, 20, 24, 24, 28, 28, 32, 32]
N_EST = 45

t0 = time.time()
Xtr = prepare(train)
models = []
for s, depth in enumerate(BAG_DEPTHS):
    m = xgb.XGBClassifier(
        n_estimators=N_EST,
        max_depth=depth,
        learning_rate=0.05,
        colsample_bytree=0.8,
        tree_method="hist",
        max_bin=128,
        enable_categorical=True,
        random_state=1000 + s,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, y, sample_weight=w)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
