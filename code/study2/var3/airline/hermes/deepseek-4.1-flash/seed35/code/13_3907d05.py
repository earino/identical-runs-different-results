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
# Month/DayofMonth/DayOfWeek are kept as numeric ordinals in prepare() instead (see there)
CAL_CAT = ["Month", "DayofMonth", "DayOfWeek"]
cat_cols = [c for c in cat_cols if c not in CAL_CAT]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# congestion frequencies (fit on train only): share of scheduled departures per airport/hour, carrier/hour
_train_hour = (train["DepTime"] // 100).clip(0, 23)


def _congestion(df: pd.DataFrame, a: str, b) -> pd.Series:
    return df[a].astype(str) + "|" + pd.Series(b, index=df.index).astype(str)


CF = {
    "origin_hour": _congestion(train, "Origin", _train_hour).value_counts(normalize=True).to_dict(),
    "carrier_hour": _congestion(train, "UniqueCarrier", _train_hour).value_counts(normalize=True).to_dict(),
}
# hub structure: how much of a carrier's schedule sits at this airport, and vice versa
_OC = train["Origin"].astype(str) + "|" + train["UniqueCarrier"].astype(str)
_OC_CNT = _OC.value_counts()
CF["hub_carrier_share"] = (_OC_CNT / _OC.str.split("|").str[-1].map(train["UniqueCarrier"].value_counts())).to_dict()
CF["hub_origin_share"] = (_OC_CNT / _OC.str.split("|").str[0].map(train["Origin"].value_counts())).to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hour = (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23)
    X["origin_hour_freq"] = _congestion(df, "Origin", hour).map(CF["origin_hour"]).astype(float)
    X["carrier_hour_freq"] = _congestion(df, "UniqueCarrier", hour).map(CF["carrier_hour"]).astype(float)
    oc = df["Origin"].astype(str) + "|" + df["UniqueCarrier"].astype(str)
    X["hub_carrier_share"] = oc.map(CF["hub_carrier_share"]).astype(float)
    X["hub_origin_share"] = oc.map(CF["hub_origin_share"]).astype(float)
    # ordinal re-parameterisations of the c-<n> calendar columns: numeric order lets a single split express
    # seasonality / week position, which partition splits on the raw categoricals cannot
    cnum = lambda c: pd.to_numeric(df[c].astype(str).str.slice(2), errors="coerce")
    month, dom, dow = cnum("Month"), cnum("DayofMonth"), cnum("DayOfWeek")
    X["month_num"] = month
    X["dom_num"] = dom
    X["dow_num"] = dow
    X["day_of_year"] = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)[
        month.fillna(1).astype(int).clip(1, 12)
    ] + dom
    X["is_weekend"] = (dow >= 6).astype(float)
    # travel-peak / holiday structure inferred from (month, day-of-month, day-of-week): fixed-date and
    # floating US holidays, so the pattern transfers to another year instead of being tied to a day number
    m, d, w = month, dom, dow
    holidays = [
        (m == 1) & (d <= 2),                              # New Year
        (m == 12) & (d >= 31),                            # New Year's Eve
        (m == 7) & (d >= 3) & (d <= 5),                   # July 4th
        (m == 11) & (w == 4) & (d >= 22) & (d <= 28),     # Thanksgiving (4th Thursday)
        (m == 9) & (w == 1) & (d <= 7),                   # Labor Day (1st Monday)
        (m == 5) & (w == 1) & (d >= 25),                  # Memorial Day (last Monday)
        (m == 12) & (d >= 24) & (d <= 26),                # Christmas
    ]
    peak_weeks = [
        (m == 11) & (d >= 20) & (d <= 30),                # Thanksgiving travel week
        (m == 12) & (d >= 20),                            # Christmas/New Year travel week
        (m == 12) & (d <= 2),
        (m == 7) & (d >= 1) & (d <= 8),                   # July 4th week
        (m == 9) & (d <= 8),                              # Labor Day week
        (m == 5) & (d >= 20),                             # Memorial Day week
        (m == 1) & (d <= 5),
    ]
    X["is_holiday"] = np.logical_or.reduce([h.to_numpy() for h in holidays]).astype(float)
    X["is_peak_week"] = np.logical_or.reduce([p.to_numpy() for p in peak_weeks]).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: average of a hyperparameter-diverse bag of XGBoost fits --------------
# (seed, max_depth, min_child_weight, subsample, colsample_bytree, learning_rate)
MEMBERS = [
    (42, 5, 5, 0.9, 0.8, 0.03),
    (101, 6, 5, 0.9, 0.8, 0.03),
    (202, 5, 5, 0.9, 0.8, 0.03),
    (303, 7, 20, 0.8, 0.6, 0.03),
    (404, 5, 5, 0.7, 1.0, 0.03),
    (505, 6, 10, 0.9, 0.7, 0.02),
    (606, 5, 10, 0.8, 0.6, 0.03),
    (707, 7, 20, 0.9, 0.7, 0.02),
]


def _make_model(seed: int, depth: int, mcw: float, sub: float, cols: float, lr: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=depth,
        learning_rate=lr,
        min_child_weight=mcw,
        subsample=sub,
        colsample_bytree=cols,
        reg_lambda=5.0,
        tree_method="hist",
        num_parallel_tree=2,
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        early_stopping_rounds=50,
        eval_metric="auc",
    )


t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = []
for cfg in MEMBERS:
    m = _make_model(*cfg)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  best_iterations={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
