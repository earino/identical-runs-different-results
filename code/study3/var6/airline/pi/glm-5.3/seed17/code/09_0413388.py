"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}

# --- schedule-volume statistics (fit on train only, no target involved) ---------
_hour_tr = (pd.to_numeric(train["DepTime"], errors="coerce") // 100).fillna(-1).astype(int).astype(str)
_hb_key_tr = (pd.to_numeric(train["DepTime"], errors="coerce") // 100 // 3).fillna(-1).astype(int).astype(str)
VOL_MAPS = {
    "origin_hour": (train["Origin"].astype(str) + "_" + _hour_tr).value_counts().to_dict(),
    "dest_hour": (train["Dest"].astype(str) + "_" + _hour_tr).value_counts().to_dict(),
    "origin_day": train["Origin"].astype(str).value_counts().to_dict(),
    "dest_day": train["Dest"].astype(str).value_counts().to_dict(),
    "route_day": (train["Origin"].astype(str) + "-" + train["Dest"].astype(str)).value_counts().to_dict(),
    "carrier_hour": (train["UniqueCarrier"].astype(str) + "_" + _hour_tr).value_counts().to_dict(),
    "origin_hb": (train["Origin"].astype(str) + "_" + _hb_key_tr).value_counts().to_dict(),
    "dest_hb": (train["Dest"].astype(str) + "_" + _hb_key_tr).value_counts().to_dict(),
}


def _cnum(s: pd.Series) -> pd.Series:
    """c-<n> string column -> numeric (leave NaN as NaN)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    X["month"] = month
    X["day"] = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["dow"] = dow
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_time"] = dep
    hour = (dep // 100).fillna(0).astype(int)
    minute = (dep % 100).fillna(0).astype(int)
    X["hour"] = hour
    X["minute"] = minute
    depmin = hour * 60 + minute
    X["depmin"] = depmin
    tod = depmin / 1440.0
    X["sin_tod"] = np.sin(2 * np.pi * tod)
    X["cos_tod"] = np.cos(2 * np.pi * tod)
    # phase-aligned day cycle: 0 at ~5am, the daily delay minimum (delay accumulates through the operating day)
    dmin = (depmin - 300) % 1440
    X["depmin_shift"] = dmin
    tod2 = dmin / 1440.0
    X["sin_tod_shift"] = np.sin(2 * np.pi * tod2)
    X["cos_tod_shift"] = np.cos(2 * np.pi * tod2)
    X["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    X["sin_dow"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["cos_dow"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    # schedule-volume statistics from TRAIN 2005 (stable structural facts, no target involved)
    oh = df["Origin"].astype(str) + "_" + hour.astype(str)
    dh = df["Dest"].astype(str) + "_" + hour.astype(str)
    hb = (hour // 3).astype(str)
    ohb = df["Origin"].astype(str) + "_" + hb
    dhb = df["Dest"].astype(str) + "_" + hb
    X["vol_origin_hour"] = oh.map(VOL_MAPS["origin_hour"]).fillna(0)
    X["vol_dest_hour"] = dh.map(VOL_MAPS["dest_hour"]).fillna(0)
    X["vol_origin_day"] = df["Origin"].astype(str).map(VOL_MAPS["origin_day"]).fillna(0)
    X["vol_dest_day"] = df["Dest"].astype(str).map(VOL_MAPS["dest_day"]).fillna(0)
    X["vol_route_day"] = (df["Origin"].astype(str) + "-" + df["Dest"].astype(str)).map(VOL_MAPS["route_day"]).fillna(0)
    X["vol_carrier_hour"] = (df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)).map(VOL_MAPS["carrier_hour"]).fillna(0)
    X["vol_origin_hb"] = ohb.map(VOL_MAPS["origin_hb"]).fillna(0)
    X["vol_dest_hb"] = dhb.map(VOL_MAPS["dest_hb"]).fillna(0)
    X["share_origin_hour"] = X["vol_origin_hour"] / (X["vol_origin_day"] + 1)
    X["share_dest_hour"] = X["vol_dest_hour"] / (X["vol_dest_day"] + 1)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of 3 XGB configs with different seeds/depths; average their probabilities.
CONFIGS = [
    dict(n_estimators=200, max_depth=8, learning_rate=0.1, colsample_bytree=0.8, subsample=0.8, min_child_weight=10),
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, colsample_bytree=0.7, subsample=0.9, min_child_weight=10),
    dict(n_estimators=120, max_depth=10, learning_rate=0.1, colsample_bytree=0.9, subsample=0.7, min_child_weight=10),
    dict(n_estimators=800, max_depth=4, learning_rate=0.03, colsample_bytree=0.8, subsample=0.85, min_child_weight=10),
    dict(n_estimators=300, max_depth=7, learning_rate=0.07, colsample_bytree=0.6, subsample=0.75, min_child_weight=25),
    dict(n_estimators=80, max_depth=12, learning_rate=0.15, colsample_bytree=1.0, subsample=0.8, min_child_weight=5),
]
X_all, y_all = prepare(train), to_y(train)
models = []
t0 = time.time()
for i, cfg in enumerate(CONFIGS * 2):  # each config twice with different seeds
    m = xgb.XGBClassifier(
        **cfg,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 7 * i,
        n_jobs=N_JOBS,
    )
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")
imp = pd.Series(models[0].feature_importances_, index=X_all.columns).sort_values(ascending=False)
print("feature importance (model 0):\n" + imp.round(4).to_string())


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
