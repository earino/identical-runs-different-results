"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives inside prepare(); every fitted table is estimated on `train` only.
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


def _cnum(series: pd.Series) -> pd.Series:
    """'c-12' -> 12; anything unparsable -> NaN."""
    return pd.to_numeric(series.astype(str).str.replace("c-", "", regex=False), errors="coerce")


# --- fitted tables (training split only) ---------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}

_route_train = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
freq_tables = {
    "UniqueCarrier": train["UniqueCarrier"].astype(str).value_counts(),
    "Origin": train["Origin"].astype(str).value_counts(),
    "Dest": train["Dest"].astype(str).value_counts(),
    "route": _route_train.value_counts(),
}

NUM_COLS = [
    "Month_n", "DayofMonth_n", "DayOfWeek_n", "is_weekend",
    "dep_hour", "dep_minute", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_red_eye",
    "Distance", "log_distance",
    "carrier_freq", "origin_freq", "dest_freq", "route_freq",
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw DataFrame -> model matrix."""
    X = pd.DataFrame(index=df.index)

    X["Month_n"] = _cnum(df["Month"])
    X["DayofMonth_n"] = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["DayOfWeek_n"] = dow
    X["is_weekend"] = dow.isin([6.0, 7.0]).astype(int)

    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 23)
    X["dep_hour"] = hour
    X["dep_minute"] = dt % 100
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["is_red_eye"] = ((hour <= 5) | (hour >= 22)).astype(int)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_distance"] = np.log1p(dist)

    carrier = df["UniqueCarrier"].astype(str)
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    route = origin + "_" + dest
    X["carrier_freq"] = carrier.map(freq_tables["UniqueCarrier"]).fillna(0.0)
    X["origin_freq"] = origin.map(freq_tables["Origin"]).fillna(0.0)
    X["dest_freq"] = dest.map(freq_tables["Dest"]).fillna(0.0)
    X["route_freq"] = route.map(freq_tables["route"]).fillna(0.0)

    for c, s in (("UniqueCarrier", carrier), ("Origin", origin), ("Dest", dest)):
        X[c] = pd.Categorical(s, categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# The 2005 -> 2006 shift is large: an internal 2005 holdout keeps improving with
# capacity while 2006 AUC degrades, so this model is deliberately small and heavily
# regularized (no early stopping; full training data is used).
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.03,
    min_child_weight=100,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
