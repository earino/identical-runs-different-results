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
CAL = ("Month", "DayofMonth", "DayOfWeek")
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000 and c not in CAL]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# congestion proxies: how many scheduled departures share the same airport/hour (and carrier/day) in train
_h_tr = ((train["DepTime"] // 100) % 24).astype(str)
_dow_tr = train["DayOfWeek"].astype(str)
_orig_tr = train["Origin"].astype(str)
_dest_tr = train["Dest"].astype(str)
_carr_tr = train["UniqueCarrier"].astype(str)
_route_tr = _orig_tr + "_" + _dest_tr
FREQ = {
    "origin_hour": (_orig_tr + "_" + _h_tr).value_counts(),
    "dest_hour": (_dest_tr + "_" + _h_tr).value_counts(),
    "carrier_dow": (_carr_tr + "_" + _dow_tr).value_counts(),
    "carrier_hour": (_carr_tr + "_" + _h_tr).value_counts(),
    "route": _route_tr.value_counts(),
    "route_carrier": (_route_tr + "_" + _carr_tr).value_counts(),
}


def _num(s: pd.Series) -> pd.Series:
    # "c-4" -> 4 ; plain numeric -> itself
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    dep = _num(X["DepTime"])
    hour = (dep // 100) % 24
    X["dep_hour"] = hour
    X["dep_minutes"] = hour * 60 + (dep % 100)
    month = _num(df["Month"])
    dow = _num(df["DayOfWeek"])
    dom = pd.to_numeric(df["DayofMonth"], errors="coerce")
    X["month_num"] = month
    X["dow_num"] = dow
    X["dom_num"] = dom
    X["is_weekend"] = (dow >= 6).astype(int)
    X["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)

    orig = X["Origin"].astype(str)
    dest = X["Dest"].astype(str)
    hs = hour.astype(str)
    X["origin_hour"] = (orig + "_" + hs).map(FREQ["origin_hour"]).fillna(0).astype(float)
    X["dest_hour"] = (dest + "_" + hs).map(FREQ["dest_hour"]).fillna(0).astype(float)
    carr = X["UniqueCarrier"].astype(str)
    X["carrier_dow"] = (carr + "_" + dow.astype(str)).map(FREQ["carrier_dow"]).fillna(0).astype(float)
    X["carrier_hour"] = (carr + "_" + hs).map(FREQ["carrier_hour"]).fillna(0).astype(float)
    X["route_freq"] = (orig + "_" + dest).map(FREQ["route"]).fillna(0).astype(float)
    X["route_carrier"] = (orig + "_" + dest + "_" + carr).map(FREQ["route_carrier"]).fillna(0).astype(float)
    X["dist_log"] = np.log1p(X["Distance"].astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(depth: int, n_estimators: int, lr: float, seed: int = SEED,
               subsample: float = 1.0, colsample: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=depth,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=colsample,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


SPECS = [
    (1, 3000, 0.03),
    (2, 1000, 0.02),
    (3, 1600, 0.01),
    (4, 600, 0.02),
    (5, 300, 0.02),
    (6, 200, 0.02),
    (7, 150, 0.02),
    (8, 120, 0.02),
]
X_train = prepare(train)
y_train = to_y(train)

t0 = time.time()
models = [make_model(d, n, lr).fit(X_train, y_train) for d, n, lr in SPECS]
for i, (d, n, lr) in enumerate(SPECS):
    models.append(make_model(d, n, lr, seed=SEED + 100 + i, subsample=0.8, colsample=0.7).fit(X_train, y_train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
