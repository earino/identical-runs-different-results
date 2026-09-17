"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# statistics computed on TRAIN only
freq_maps = {}
for c in ("UniqueCarrier", "Origin", "Dest"):
    freq_maps[c] = train[c].value_counts().to_dict()
_route = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str))
freq_maps["route"] = _route.value_counts().to_dict()
_rd = train[["Distance"]].copy()
_rd["route"] = _route.values
route_mean_dist = _rd.groupby("route")["Distance"].mean().to_dict()
_hr = train["DepTime"] // 100
_oh = train.groupby([train["Origin"], _hr]).size().to_dict()
_dh = train.groupby([train["Dest"], _hr]).size().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    # calendar columns are "c-<n>" strings -> numeric
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        X[c] = X[c].astype(str).str.replace("c-", "", regex=False).astype(int)
    # time-of-day features from DepTime (hhmm)
    hour = X["DepTime"] // 100
    minute = X["DepTime"] % 100
    tod = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    X["sin_tod2"] = np.sin(4 * np.pi * tod / 1440.0)
    X["cos_tod2"] = np.cos(4 * np.pi * tod / 1440.0)
    X["hour_cat"] = pd.Categorical(hour.astype(int), categories=range(24))
    X["log_distance"] = np.log1p(X["Distance"])
    # busyness features (maps fit on train)
    route = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    X["route_freq"] = route.map(freq_maps["route"]).fillna(0.0)
    X["origin_freq"] = X["Origin"].map(freq_maps["Origin"]).fillna(0.0)
    X["dest_freq"] = X["Dest"].map(freq_maps["Dest"]).fillna(0.0)
    X["carrier_freq"] = X["UniqueCarrier"].map(freq_maps["UniqueCarrier"]).fillna(0.0)
    rmd = route.map(route_mean_dist)
    X["rel_dist"] = X["Distance"] / rmd  # NaN where route unseen
    # congestion proxies: same-origin / same-dest traffic in the same hour (train stats)
    oh_key = list(zip(X["Origin"].astype(str), X["hour"].astype(int)))
    X["origin_hour_freq"] = pd.Series(oh_key).map(_oh).fillna(0.0)
    dh_key = list(zip(X["Dest"].astype(str), X["hour"].astype(int)))
    X["dest_hour_freq"] = pd.Series(dh_key).map(_dh).fillna(0.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=24,
    learning_rate=0.007,
    early_stopping_rounds=250,
    subsample=0.5,
    colsample_bytree=0.5,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))])
print(f"Training time: {time.time() - t0:.1f}s  best_iter={getattr(model, 'best_iteration', '?')}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
