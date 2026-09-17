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

# count encodings (frequency of each key in train)
cnt_maps = {n: _KEYS_TRAIN[TE_SPEC[n][0]].value_counts() for n in TE_SPEC}

# --- feature switches (module level so prepare() stays the single code path) ----
USE_TIME = False
USE_COUNTS = False
ACTIVE_TE = ["Origin", "Dest", "UniqueCarrier", "Origin_hour"]


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
        dep = df["DepTime"].astype(float).to_numpy()
        dep = np.where(dep >= 2400, dep - 2400.0, dep)
        hour = np.floor(dep / 100.0)
        tod = hour * 60.0 + (dep - hour * 100.0)
        mon = df["Month"].astype(str).str.slice(2).astype(float).to_numpy()
        dow = df["DayOfWeek"].astype(str).str.slice(2).astype(float).to_numpy()
        X["dep_hour"] = hour
        X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
        X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
        X["month_sin"] = np.sin(2 * np.pi * mon / 12.0)
        X["month_cos"] = np.cos(2 * np.pi * mon / 12.0)
        X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
        X["is_weekend"] = (dow >= 6).astype(float)
        X["log_distance"] = np.log1p(df["Distance"].astype(float).to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def build(df, use_time, use_counts, te_cols):
    global USE_TIME, USE_COUNTS, ACTIVE_TE
    USE_TIME, USE_COUNTS, ACTIVE_TE = use_time, use_counts, te_cols
    X = prepare(df)
    if df is train:
        for n in te_cols:
            X[f"te_{n}"] = oof_te[n]
    USE_TIME, USE_COUNTS = False, False
    return X


ytr, yev = to_y(train), to_y(evald)
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
REF = dict(max_depth=4, learning_rate=0.02, n_estimators=1200, reg_alpha=10.0)
CORE = ["Origin", "Dest", "UniqueCarrier", "Origin_hour"]

# --- DIAGNOSTIC: encoding layer under L1 (smoothing & new keys) ------------------
CANDS = {
    "ref t+c n2000": (True, True, CORE, dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
    "hour k10": (True, True, ["Origin", "Dest", "UniqueCarrier", "Origin_hour_lo"], dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
    "hour k100": (True, True, ["Origin", "Dest", "UniqueCarrier", "Origin_hour_hi"], dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
    "+wk_hour": (True, True, CORE + ["Origin_wk_hour"], dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
    "+Carrier_origin": (True, True, CORE + ["Carrier_origin"], dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
    "+Hour": (True, True, CORE + ["Hour"], dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
    "+Route": (True, True, CORE + ["Route"], dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
    "+Origin_month": (True, True, CORE + ["Origin_month"], dict(max_depth=4, n_estimators=2000, reg_alpha=10.0)),
}
for _label, (_t, _c, _te, _pp) in CANDS.items():
    _A, _B = build(train, _t, _c, _te), build(evald, _t, _c, _te)
    _m = xgb.XGBClassifier(random_state=SEED, **{**BASE, **REF, **_pp})
    _m.fit(_A, ytr)
    print(f"diag {_label:24s} nfeat={_A.shape[1]:3d} eval_auc={roc_auc_score(yev, _m.predict_proba(_B)[:, 1]):.4f}")

# --- model: time + count features + L1-regularized reference -------------------
Xtr = build(train, True, True, CORE)
Xev = build(evald, True, True, CORE)
USE_TIME, USE_COUNTS, ACTIVE_TE = True, True, CORE
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
