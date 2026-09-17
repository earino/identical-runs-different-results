"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature definition ---------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["Distance"]
# category levels fitted on training data only; unseen levels at predict time become NaN
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
# traffic-volume features: how busy an airport / carrier is (stable across years, unlike route-level effects)
_h_tr = (train["DepTime"].astype(int) // 100).astype(str)
_h_ev = (evald["DepTime"].astype(int) // 100).astype(str)
FREQ = {
    "orig_freq": train["Origin"].value_counts(),
    "dest_freq": train["Dest"].value_counts(),
    "carrier_freq": train["UniqueCarrier"].value_counts(),
    "orig_routes": (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts().groupby(
        (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).str.slice(0, 3)).sum(),
    "orig_hour": (train["Origin"].astype(str) + "|" + _h_tr).value_counts(),
    "carrier_hour": (train["UniqueCarrier"].astype(str) + "|" + _h_tr).value_counts(),
    "dest_hour": (train["Dest"].astype(str) + "|" + _h_tr).value_counts(),
}


def _cyc(v, period):
    a = 2.0 * np.pi * v / period
    return np.sin(a), np.cos(a)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # calendar: "c-<n>" strings -> integers
    month = df["Month"].str.slice(2).astype(int)
    dom = df["DayofMonth"].str.slice(2).astype(int)
    dow = df["DayOfWeek"].str.slice(2).astype(int)
    X["Month"] = month
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    ms, mc = _cyc(month - 1, 12.0)
    X["month_sin"], X["month_cos"] = ms, mc
    ds, dc = _cyc(dom - 1, 31.0)
    X["dom_sin"], X["dom_cos"] = ds, dc
    ws, wc = _cyc(dow - 1, 7.0)
    X["dow_sin"], X["dow_cos"] = ws, wc
    # scheduled departure time (hhmm; values >= 2400 roll into the next day)
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    tod = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_tod"] = tod
    ts, tc = _cyc(tod, 1440.0)
    X["tod_sin"], X["tod_cos"] = ts, tc
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    X["orig_freq"] = df["Origin"].map(FREQ["orig_freq"]).astype(float)
    X["dest_freq"] = df["Dest"].map(FREQ["dest_freq"]).astype(float)
    X["carrier_freq"] = df["UniqueCarrier"].map(FREQ["carrier_freq"]).astype(float)
    X["orig_routes"] = df["Origin"].map(FREQ["orig_routes"]).astype(float)
    X["orig_hour"] = (df["Origin"].astype(str) + "|" + hour.astype(str)).map(FREQ["orig_hour"]).astype(float)
    X["carrier_hour"] = (df["UniqueCarrier"].astype(str) + "|" + hour.astype(str)).map(FREQ["carrier_hour"]).astype(float)
    X["dest_hour"] = (df["Dest"].astype(str) + "|" + hour.astype(str)).map(FREQ["dest_hour"]).astype(float)
    # share of an airport's / carrier's traffic that leaves in this hour (season/bank structure, year-stable)
    X["orig_hour_frac"] = X["orig_hour"] / X["orig_freq"]
    X["carrier_hour_frac"] = X["carrier_hour"] / X["carrier_freq"]
    X["dest_hour_frac"] = X["dest_hour"] / X["dest_freq"]
    cat_levels = CAT_LEVELS
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
# bag of XGBoost models: different seeds and mildly different sampling/regularisation.
# averaging decorrelated fits reduces variance on the drifted (2006) evaluation data.
Xtr = prepare(train)
ytr = to_y(train)

MODELS = []
t0 = time.time()
for i, seed in enumerate((42, 7, 2024)):
    m = xgb.XGBClassifier(
        n_estimators=900,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=30 + 10 * i,
        subsample=0.85,
        colsample_bytree=0.7 - 0.05 * i,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)  # every member sees all rows; tree count fixed from the early-stopping study
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
