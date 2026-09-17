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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# interaction categorical levels (train only)
_hh = pd.Series(np.floor(np.asarray(pd.to_numeric(train["DepTime"], errors="coerce"), dtype="float64") / 100.0) % 24, index=train.index)
carrier_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_h" + _hh.astype("Int64").astype(str)).dropna().unique()))
_dep_min = np.asarray(pd.to_numeric(train["DepTime"], errors="coerce"), dtype="float64")
_dep_hh = np.floor(_dep_min / 100.0)
_min_of_day = _dep_hh * 60.0 + (_dep_min - 100.0 * _dep_hh)
_slot30 = pd.Series(np.floor(_min_of_day / 30.0), index=train.index).astype("Int64").astype(str)
carrier_slot_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_s" + _slot30).dropna().unique()))
origin_slot30_levels = pd.Index(sorted((train["Origin"].astype(str) + "_s" + _slot30).dropna().unique()))

# --- route distance stats, fit on train ONLY ------------------------------------
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_route_cnt = _route.value_counts()
_route_dist = train["Distance"].groupby(_route).mean()
_route_dist_std = train["Distance"].groupby(_route).std().fillna(0.0)
GLOBAL_DIST = float(train["Distance"].mean())


def route_stats(df):
    r = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    cnt = r.map(_route_cnt).fillna(0.0).to_numpy(dtype="float64")
    avg = r.map(_route_dist).fillna(GLOBAL_DIST).to_numpy(dtype="float64")
    std = r.map(_route_dist_std).fillna(0.0).to_numpy(dtype="float64")
    return cnt, avg, std

# raw column names usable at predict time
RAW = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
RAW = [c for c in RAW if c in train.columns and c != TARGET]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # --- engineered numeric features from raw columns ---
    def g(col):
        if col in df.columns:
            return np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype="float64")
        return np.full(len(df), np.nan)

    mon = g("Month")
    dom = g("DayofMonth")
    dow = g("DayOfWeek")
    dep = g("DepTime")
    dist = g("Distance")

    X["f_dep_time"] = dep  # raw hhmm integer
    X["f_dep_hour"] = np.floor(dep / 100) % 24
    X["f_dep_minute"] = dep % 100
    X["f_dep_frac"] = (dep % 2400) / 24.0  # hhmm as fraction of day (careful: 2400 wraps)
    X["f_dep_norm"] = dep / 2400.0
    X["f_dep_sind"] = np.sin(2 * np.pi * dep / 2400.0)
    X["f_dep_cosd"] = np.cos(2 * np.pi * dep / 2400.0)
    X["f_dom"] = dom
    X["f_dow"] = dow
    X["f_month"] = mon
    X["f_dist"] = dist
    X["f_dist_log"] = np.log1p(dist)
    # minute-of-day: a cleaner continuous clock than hhmm
    hh = np.floor(dep / 100.0)
    X["f_minOfDay"] = hh * 60.0 + (dep - 100.0 * hh)
    # seasonal encodings (month/dom/dow raw strings coerced earlier are NaN, so derive numerics properly)
    def _strip_c(s):
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            return np.asarray(pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce"), dtype="float64")
        return np.asarray(pd.to_numeric(s, errors="coerce"), dtype="float64")

    mon_n = _strip_c(df["Month"]) if "Month" in df.columns else np.full(len(df), np.nan)
    dom_n = _strip_c(df["DayofMonth"]) if "DayofMonth" in df.columns else np.full(len(df), np.nan)
    dow_n = _strip_c(df["DayOfWeek"]) if "DayOfWeek" in df.columns else np.full(len(df), np.nan)
    X["f_mon_n"] = mon_n
    X["f_dom_n"] = dom_n
    X["f_dow_n"] = dow_n
    X["f_mon_sin"] = np.sin(2 * np.pi * mon_n / 12.0)
    X["f_mon_cos"] = np.cos(2 * np.pi * mon_n / 12.0)
    doy = 30.4 * (mon_n - 1) + dom_n
    X["f_doy"] = doy
    X["f_doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["f_doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    # interaction categorical: carrier x hour-of-day
    if "UniqueCarrier" in df.columns:
        hhs = pd.Series(hh % 24, index=df.index).astype("Int64").astype(str)
        hc = df["UniqueCarrier"].astype(str) + "_h" + hhs
        X["f_carrier_hour"] = pd.Categorical(hc, categories=carrier_hour_levels)
    # carrier x 30-min slot
    if "UniqueCarrier" in df.columns:
        hhs30 = pd.Series(hh * 60.0 + (dep - 100.0 * hh), index=df.index)
        s30 = (np.floor(hhs30 / 30.0)).astype("Int64").astype(str)
        sc = df["UniqueCarrier"].astype(str) + "_s" + s30
        X["f_carrier_slot30"] = pd.Categorical(sc, categories=carrier_slot_levels)
    # route stats (train-only)
    if "Origin" in df.columns and "Dest" in df.columns:
        cnt, avg, std = route_stats(df)
        X["f_route_count"] = cnt
        X["f_route_avgdist"] = avg
        X["f_route_stddist"] = std
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed ensemble (XGBoost only, allowed) ------------------------------
N_MODELS = 14
t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
models = []
CFG = [
    dict(max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0),
    dict(max_depth=7, subsample=0.7, colsample_bytree=0.7, min_child_weight=10, reg_lambda=3.0),
    dict(max_depth=9, subsample=0.9, colsample_bytree=0.6, min_child_weight=3, reg_lambda=1.0),
    dict(max_depth=6, subsample=0.75, colsample_bytree=0.9, min_child_weight=8, reg_lambda=2.0),
    dict(max_depth=10, subsample=0.85, colsample_bytree=0.75, min_child_weight=6, reg_lambda=4.0),
    dict(max_depth=8, subsample=0.9, colsample_bytree=0.9, min_child_weight=5, reg_lambda=1.5),
    dict(max_depth=7, subsample=0.8, colsample_bytree=0.6, min_child_weight=7, reg_lambda=2.5),
    dict(max_depth=9, subsample=0.7, colsample_bytree=0.8, min_child_weight=4, reg_lambda=3.5),
    dict(max_depth=6, subsample=0.85, colsample_bytree=0.7, min_child_weight=12, reg_lambda=1.0),
    dict(max_depth=11, subsample=0.75, colsample_bytree=0.85, min_child_weight=5, reg_lambda=5.0),
    dict(max_depth=0, max_leaves=64, grow_policy="lossguide", subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0),
    dict(max_depth=0, max_leaves=96, grow_policy="lossguide", subsample=0.85, colsample_bytree=0.7, min_child_weight=8, reg_lambda=3.0),
    dict(max_depth=0, max_leaves=48, grow_policy="lossguide", subsample=0.75, colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.5),
    dict(max_depth=0, max_leaves=128, grow_policy="lossguide", subsample=0.9, colsample_bytree=0.75, min_child_weight=6, reg_lambda=2.5),
]
for k in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + k,
        n_jobs=N_JOBS,
        **CFG[k],
    )
    m.fit(Xtr, ytr)
    models.append(m)
model = models[0]
print(f"Training time: {time.time() - t0:.1f}s for {N_MODELS} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
