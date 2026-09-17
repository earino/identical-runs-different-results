"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  3. All feature engineering lives in prepare(); every statistic it uses is fit on data/train.csv only.
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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(Y_TRAIN.mean())


def add_keys(df: pd.DataFrame) -> dict:
    """Raw -> string keys for target encoding (no train statistics involved)."""
    dep = df["DepTime"].astype(float)
    dep = dep.where(dep < 2400, dep - 2400.0)
    hour = (dep // 100.0).astype(int)
    hour_s = hour.astype(str)
    origin, dest = df["Origin"].astype(str), df["Dest"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(int)
    return {
        "Origin": origin,
        "Dest": dest,
        "UniqueCarrier": carrier,
        "Origin_hour": origin + "|" + hour_s,
        "Origin_hour_lo": origin + "|" + hour_s,
        "Origin_hour_hi": origin + "|" + hour_s,
        "Dest_hour": dest + "|" + hour_s,
        "Origin_dow": origin + "|" + dow.astype(str),
        "Origin_month": origin + "|" + df["Month"].astype(str),
        "Origin_wk_hour": origin + "|" + (dow >= 6).astype(int).astype(str) + "|" + hour_s,
        "Carrier_origin": carrier + "|" + origin,
        "Hour": hour_s,
        "Route": origin + "|" + dest,
    }


# --- target encoding: smoothed means, fit on TRAIN ONLY ------------------------
TE_SPEC = {
    "Origin": ("Origin", 50.0), "Dest": ("Dest", 50.0), "UniqueCarrier": ("UniqueCarrier", 50.0),
    "Origin_hour": ("Origin_hour", 30.0),
    "Origin_hour_lo": ("Origin_hour_lo", 10.0), "Origin_hour_hi": ("Origin_hour_hi", 100.0),
    "Dest_hour": ("Dest_hour", 100.0), "Origin_dow": ("Origin_dow", 100.0),
    "Origin_month": ("Origin_month", 100.0), "Origin_wk_hour": ("Origin_wk_hour", 100.0),
    "Carrier_origin": ("Carrier_origin", 100.0), "Hour": ("Hour", 100.0), "Route": ("Route", 100.0),
}
_KEYS_TRAIN = add_keys(train)


def _te_stats(keys, y, k):
    s = pd.Series(y, index=keys.index).groupby(keys.to_numpy()).agg(["sum", "count"])
    return ((s["sum"] + PRIOR * k) / (s["count"] + k)).to_dict()


te_maps, oof_te = {}, {}
for _n, (_key, _k) in TE_SPEC.items():
    _ks = _KEYS_TRAIN[_key]
    te_maps[_n] = _te_stats(_ks, Y_TRAIN, _k)
    _oof = np.full(len(train), PRIOR)
    for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
        _m = _te_stats(_ks.iloc[_tr], Y_TRAIN[_tr], _k)
        _oof[_va] = _ks.iloc[_va].map(_m).fillna(PRIOR).to_numpy()
    oof_te[_n] = _oof

cnt_maps = {n: _KEYS_TRAIN[TE_SPEC[n][0]].value_counts() for n in TE_SPEC}
KC = {k: v for k, v in _KEYS_TRAIN.items()}

TIME_FEATURES = {"dep_hour", "tod_sin", "tod_cos", "month_sin", "month_cos",
                 "dow_sin", "dow_cos", "is_weekend", "log_distance", "dep_minute"}
USE_TIME = True
USE_COUNTS = True
ACTIVE_TE = ["Origin", "Dest", "UniqueCarrier", "Origin_hour"]


def _time_block(df: pd.DataFrame) -> dict:
    dep = df["DepTime"].astype(float)
    dep = dep.where(dep < 2400, dep - 2400.0)
    hour = (dep // 100.0)
    tod = hour * 60.0 + (dep - hour * 100.0)
    mon = df["Month"].astype(str).str.slice(2).astype(float)
    dow = df["DayOfWeek"].astype(str).str.slice(2).astype(float)
    return {
        "dep_hour": hour,
        "dep_minute": dep - hour * 100.0,
        "tod_sin": np.sin(2 * np.pi * tod / 1440.0),
        "tod_cos": np.cos(2 * np.pi * tod / 1440.0),
        "month_sin": np.sin(2 * np.pi * mon / 12.0),
        "month_cos": np.cos(2 * np.pi * mon / 12.0),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 6).astype(float),
        "log_distance": np.log1p(df["Distance"].astype(float)),
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    keys = add_keys(df)
    for n in ACTIVE_TE:
        X[f"te_{n}"] = keys[TE_SPEC[n][0]].map(te_maps[n]).fillna(PRIOR).to_numpy()
        if USE_COUNTS:
            X[f"cnt_{n}"] = keys[TE_SPEC[n][0]].map(cnt_maps[n]).fillna(0).to_numpy()
    if USE_TIME:
        block = _time_block(df)
        for k in TIME_FEATURES:
            X[k] = block[k].to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def build(df, te_cols, use_time=True, use_counts=True, time_features=None):
    global USE_TIME, USE_COUNTS, ACTIVE_TE, TIME_FEATURES
    _st, _sc, _sf = TIME_FEATURES, USE_TIME, USE_COUNTS
    ACTIVE_TE = te_cols
    USE_TIME = use_time
    USE_COUNTS = use_counts
    if time_features is not None:
        TIME_FEATURES = set(time_features)
    X = prepare(df)
    if df is train:
        for n in te_cols:
            X[f"te_{n}"] = oof_te[n]
    TIME_FEATURES, USE_TIME, USE_COUNTS = _st, True, True
    return X


ytr, yev = to_y(train), to_y(evald)
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
REF = dict(max_depth=4, learning_rate=0.02, n_estimators=2000, reg_alpha=10.0)
CORE = ["Origin", "Dest", "UniqueCarrier", "Origin_hour"]
T_ALL = sorted(TIME_FEATURES)
T_MIN = T_ALL + ["dep_minute"]

# --- DIAGNOSTIC: minute feature + time-block ablations -------------------------
CANDS = {
    "ref": (CORE, True, True, T_ALL),
    "+dep_minute": (CORE, True, True, T_MIN),
    "-minute block": (CORE, True, True, ["dep_hour", "is_weekend", "log_distance"]),
    "-log_distance": (CORE, True, True, [f for f in T_ALL if f != "log_distance"]),
    "-is_weekend": (CORE, True, True, [f for f in T_ALL if f != "is_weekend"]),
    "-cyclic month/dow": (CORE, True, True, [f for f in T_ALL if "sin" not in f and "cos" not in f]),
    "-te_Dest/UC": (["Origin", "Origin_hour"], True, True, T_ALL),
    "no counts": (CORE, True, False, T_ALL),
}
for _label, (_te, _ut, _uc, _tf) in CANDS.items():
    _A = build(train, _te, _ut, _uc, _tf)
    _B = build(evald, _te, _ut, _uc, _tf)
    _m = xgb.XGBClassifier(random_state=SEED, **{**BASE, **REF})
    _m.fit(_A, ytr)
    print(f"diag {_label:20s} nfeat={_A.shape[1]:3d} eval_auc={roc_auc_score(yev, _m.predict_proba(_B)[:, 1]):.4f}")

# --- model: time + count features + L1-regularized reference -------------------
TIME_FEATURES, USE_TIME, USE_COUNTS, ACTIVE_TE = set(T_ALL), True, True, CORE
Xtr = prepare(train)
for n in CORE:
    Xtr[f"te_{n}"] = oof_te[n]
model = xgb.XGBClassifier(random_state=SEED, **{**BASE, **REF, "n_estimators": 3000})
t0 = time.time()
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
