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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering ------------------------------------------------------
def _cnum(s: pd.Series) -> pd.Series:
    # values look like "c-4"; strip the prefix and parse the integer
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _dep_parts(df: pd.DataFrame):
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    dep = dep.where(dep > 0)
    hour = np.floor(dep / 100.0)
    minute = dep - hour * 100.0
    hour = hour.where(hour < 24, hour - 24)
    return hour, minute


cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# congestion: how many scheduled flights at this origin/dest/route in this departure hour (train counts)
_tr_hour, _ = _dep_parts(train)
_hk = _tr_hour.fillna(-1).astype(int).astype(str)
cong_maps = {
    "origin_hour_cnt": (train["Origin"].astype(str) + "|" + _hk).value_counts().to_dict(),
    "dest_hour_cnt": (train["Dest"].astype(str) + "|" + _hk).value_counts().to_dict(),
    "route_hour_cnt": (_route(train) + "|" + _hk).value_counts().to_dict(),
    "carrier_hour_cnt": (train["UniqueCarrier"].astype(str) + "|" + _hk).value_counts().to_dict(),
    "origin_cnt": train["Origin"].value_counts().to_dict(),
    "dest_cnt": train["Dest"].value_counts().to_dict(),
    "route_cnt": _route(train).value_counts().to_dict(),
    "carrier_cnt": train["UniqueCarrier"].value_counts().to_dict(),
}


# --- smoothed target encoding fit on train (OOF for the train matrix) ----------
_yte = (train[TARGET] == POSITIVE).astype(int).to_numpy()
_PRIOR = float(_yte.mean())
_SMOOTH = 20.0
_tr_dow = _cnum(train["DayOfWeek"]).fillna(-1).astype(int).astype(str)
_TE_KEYS = {
    "te_origin_hour": train["Origin"].astype(str) + "|" + _hk,
    "te_dest_hour": train["Dest"].astype(str) + "|" + _hk,
    "te_carrier_hour": train["UniqueCarrier"].astype(str) + "|" + _hk,
    "te_route_hour": _route(train) + "|" + _hk,
    "te_origin_dow": train["Origin"].astype(str) + "|" + _tr_dow,
    "te_carrier_dow": train["UniqueCarrier"].astype(str) + "|" + _tr_dow,
}


def _te_map(keys: pd.Series, y: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + _PRIOR * _SMOOTH) / (g["count"] + _SMOOTH)


_te_full = {name: _te_map(ser, _yte) for name, ser in _TE_KEYS.items()}
_te_oof = pd.DataFrame(index=train.index)
_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name, ser in _TE_KEYS.items():
    enc = np.full(len(train), _PRIOR, dtype=float)
    for tr_idx, va_idx in _kf.split(train.index):
        m = _te_map(ser.iloc[tr_idx], _yte[tr_idx])
        enc[va_idx] = ser.iloc[va_idx].map(m).fillna(_PRIOR).to_numpy()
    _te_oof[name] = enc


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    hour, minute = _dep_parts(df)
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_minutes"] = hour * 60.0 + minute
    dep_raw = pd.to_numeric(df["DepTime"], errors="coerce")
    dep_abs = (np.floor(dep_raw / 100.0)) * 60.0 + (dep_raw - np.floor(dep_raw / 100.0) * 100.0)
    X["dep_abs_minutes"] = dep_abs.where(dep_abs > 0)
    X["is_next_day"] = (dep_raw >= 2400).astype(int)
    X["dep_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["dep_cos"] = np.cos(2 * np.pi * hour / 24.0)
    X["month"] = _cnum(df["Month"])
    X["day"] = _cnum(df["DayofMonth"])
    X["dow"] = _cnum(df["DayOfWeek"])
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7.0)
    X["is_weekend"] = (X["dow"] >= 6).astype(int)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    hk = hour.fillna(-1).astype(int).astype(str)
    route = _route(df)
    X["origin_hour_cnt"] = (df["Origin"].astype(str) + "|" + hk).map(cong_maps["origin_hour_cnt"]).fillna(0)
    X["dest_hour_cnt"] = (df["Dest"].astype(str) + "|" + hk).map(cong_maps["dest_hour_cnt"]).fillna(0)
    X["route_hour_cnt"] = (route + "|" + hk).map(cong_maps["route_hour_cnt"]).fillna(0)
    X["carrier_hour_cnt"] = (df["UniqueCarrier"].astype(str) + "|" + hk).map(cong_maps["carrier_hour_cnt"]).fillna(0)
    X["origin_cnt"] = df["Origin"].map(cong_maps["origin_cnt"]).fillna(0)
    X["dest_cnt"] = df["Dest"].map(cong_maps["dest_cnt"]).fillna(0)
    X["route_cnt"] = route.map(cong_maps["route_cnt"]).fillna(0)
    X["carrier_cnt"] = df["UniqueCarrier"].map(cong_maps["carrier_cnt"]).fillna(0)
    X["origin_hour_frac"] = X["origin_hour_cnt"] / X["origin_cnt"].clip(lower=1)
    X["dest_hour_frac"] = X["dest_hour_cnt"] / X["dest_cnt"].clip(lower=1)
    X["route_hour_frac"] = X["route_hour_cnt"] / X["route_cnt"].clip(lower=1)
    X["carrier_hour_frac"] = X["carrier_hour_cnt"] / X["carrier_cnt"].clip(lower=1)
    X["te_origin_hour"] = (df["Origin"].astype(str) + "|" + hk).map(_te_full["te_origin_hour"]).fillna(_PRIOR)
    X["te_dest_hour"] = (df["Dest"].astype(str) + "|" + hk).map(_te_full["te_dest_hour"]).fillna(_PRIOR)
    X["te_carrier_hour"] = (df["UniqueCarrier"].astype(str) + "|" + hk).map(_te_full["te_carrier_hour"]).fillna(_PRIOR)
    dk = X["dow"].fillna(-1).astype(int).astype(str)
    X["te_route_hour"] = (route + "|" + hk).map(_te_full["te_route_hour"]).fillna(_PRIOR)
    X["te_origin_dow"] = (df["Origin"].astype(str) + "|" + dk).map(_te_full["te_origin_dow"]).fillna(_PRIOR)
    X["te_carrier_dow"] = (df["UniqueCarrier"].astype(str) + "|" + dk).map(_te_full["te_carrier_dow"]).fillna(_PRIOR)
    return X


def prepare_train() -> pd.DataFrame:
    X = prepare(train)
    for name in _te_full:
        X[name] = _te_oof[name].to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of xgboost variants --------------------------------------
BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    reg_lambda=3.0,
    reg_alpha=0.5,
)
CONFIGS = [
    dict(max_depth=4, n_estimators=1800, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=5, n_estimators=1200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=6, n_estimators=900, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=6, n_estimators=1200, learning_rate=0.03, subsample=0.7, colsample_bytree=0.6, min_child_weight=20),
    dict(max_depth=7, n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=8, n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10),
    dict(max_depth=9, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=20),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=64, n_estimators=700, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=128, n_estimators=500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=20),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=256, n_estimators=350, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=30),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=32, n_estimators=1000, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=192, n_estimators=450, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=15),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=384, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=40),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=512, n_estimators=250, learning_rate=0.05, subsample=0.7, colsample_bytree=0.6, min_child_weight=50),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=48, n_estimators=900, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=8),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=96, n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=12),
    dict(grow_policy="lossguide", max_depth=0, max_leaves=320, n_estimators=320, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=35),
]

X_all = prepare_train()
y_all = to_y(train)
TREE_SCALE = 0.6
t0 = time.time()
models = []
for i, cfg in enumerate(CONFIGS):
    cfg = dict(cfg)
    cfg["n_estimators"] = max(100, int(cfg["n_estimators"] * TREE_SCALE))
    m = xgb.XGBClassifier(random_state=SEED + i, **{**BASE, **cfg})
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
