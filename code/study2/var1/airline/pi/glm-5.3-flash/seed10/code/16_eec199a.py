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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]          # true categoricals, kept as such
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _c_to_int(s: pd.Series) -> pd.Series:
    # "c-<n>" -> integer n
    return s.str.slice(2).astype("int64")


# level spaces for interaction categoricals, fitted on train only
_t = train.copy()
_t["_hour"] = (_t["DepTime"].astype("int64") // 100) % 24
route_levels = pd.Index(sorted((_t["Origin"].astype(str) + "_" + _t["Dest"].astype(str)).unique()))
hr_dow_levels = pd.Index(sorted((_t["_hour"].astype(str) + "_" + _c_to_int(_t["DayOfWeek"].astype(str)).astype(str)).unique()))
car_hr_levels = pd.Index(sorted((_t["UniqueCarrier"].astype(str) + "_" + _t["_hour"].astype(str)).unique()))
# frequency encodings, fitted on train only
freq_maps = {
    "carrier_freq": _t["UniqueCarrier"].value_counts(),
    "origin_freq": _t["Origin"].value_counts(),
    "dest_freq": _t["Dest"].value_counts(),
    "route_freq": (_t["Origin"].astype(str) + "_" + _t["Dest"].astype(str)).value_counts(),
}
del _t


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # c-<n> string cols as integers
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = _c_to_int(df[c].astype(str))
    # time of day from DepTime (hhmm, may exceed 2359 -> wrap)
    dt = df["DepTime"].astype("int64")
    X["DepTime"] = dt
    hour = (dt // 100) % 24
    minute = dt % 100
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["Distance"] = df["Distance"].astype(float)
    # interaction categoricals (row-local; level spaces fitted on train only)
    X["hour_dow"] = pd.Categorical(hour.astype(str) + "_" + X["DayOfWeek"].astype(str), categories=hr_dow_levels)
    X["carrier_hour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + hour.astype(str), categories=car_hr_levels)
    X["carrier_freq"] = np.log1p(df["UniqueCarrier"].map(freq_maps["carrier_freq"]).fillna(0).astype(float))
    X["origin_freq"] = np.log1p(df["Origin"].map(freq_maps["origin_freq"]).fillna(0).astype(float))
    X["dest_freq"] = np.log1p(df["Dest"].map(freq_maps["dest_freq"]).fillna(0).astype(float))
    X["route_freq"] = np.log1p((df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).map(freq_maps["route_freq"]).fillna(0).astype(float))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int, **kw) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_child_weight=20,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(random_state=seed, **params)


models = [
    make_model(42),
    make_model(7, max_depth=5, n_estimators=300),
    make_model(21, learning_rate=0.07, n_estimators=300, min_child_weight=30),
    make_model(3, max_depth=7, min_child_weight=40),
    make_model(99, learning_rate=0.05, n_estimators=500),
    make_model(123, colsample_bytree=0.8),
    make_model(5, max_depth=5, learning_rate=0.05, n_estimators=500),
    make_model(8, min_child_weight=10),
    make_model(77, max_depth=7, n_estimators=300, learning_rate=0.08, min_child_weight=60),
    make_model(2024, subsample=0.9),
    make_model(55, max_depth=8, min_child_weight=100),
    make_model(31, learning_rate=0.15, n_estimators=150),
    make_model(64, learning_rate=0.03, n_estimators=800),
    make_model(88, max_depth=4, n_estimators=500),
    make_model(13, subsample=0.85),
    make_model(256, colsample_bytree=0.7, n_estimators=300),
]

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
for m in models:
    m.fit(X_tr, y_tr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
