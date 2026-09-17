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
        "Origin_min5": origin + "|" + hour_s + "|" + ((dep - (dep // 100.0) * 100.0) // 5).astype(int).astype(str),
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
    "Origin_min5": ("Origin_min5", 100.0),
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
# extra count-only keys (schedule density), no target encoding on them
CNT_EXTRA = {"cnt_Origin_min5": "Origin_min5", "cnt_Hour": "Hour"}
cnt_maps.update({c: _KEYS_TRAIN[k].value_counts() for c, k in CNT_EXTRA.items()})
USE_CNT_EXTRA = False
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
        "dep_min5": ((dep - hour * 100.0) // 5.0),
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
    if USE_CNT_EXTRA:
        for c, k in CNT_EXTRA.items():
            X[c] = keys[k].map(cnt_maps[c]).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def build(df, te_cols, use_time=True, use_counts=True, time_features=None, cnt_extra=False):
    global USE_TIME, USE_COUNTS, ACTIVE_TE, TIME_FEATURES, USE_CNT_EXTRA
    _st, _sc, _sf, _se = TIME_FEATURES, USE_TIME, USE_COUNTS, USE_CNT_EXTRA
    ACTIVE_TE = te_cols
    USE_TIME = use_time
    USE_COUNTS = use_counts
    USE_CNT_EXTRA = cnt_extra
    if time_features is not None:
        TIME_FEATURES = set(time_features)
    X = prepare(df)
    if df is train:
        for n in te_cols:
            X[f"te_{n}"] = oof_te[n]
    TIME_FEATURES, USE_TIME, USE_COUNTS, USE_CNT_EXTRA = _st, True, True, False
    return X


ytr, yev = to_y(train), to_y(evald)
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
REF = dict(max_depth=4, learning_rate=0.02, n_estimators=2000, reg_alpha=10.0)
CORE = ["Origin", "Dest", "UniqueCarrier", "Origin_hour"]
T_ALL = sorted(TIME_FEATURES)

# --- DIAGNOSTIC: combine 5-min TE with depth/tree variants ---------------------
T_FIN = [f for f in T_ALL if f not in ("log_distance", "is_weekend")]
M5 = CORE + ["Origin_min5"]
CANDS = {
    "d6 n1000 +min5TE": (M5, T_FIN, True, {"max_depth": 6, "n_estimators": 1000}),
    "d6 n1500 +min5TE": (M5, T_FIN, True, {"max_depth": 6, "n_estimators": 1500}),
    "d7 n800 +min5TE": (M5, T_FIN, True, {"max_depth": 7, "n_estimators": 800}),
    "d6 a12 n1000 +min5TE": (M5, T_FIN, True, {"max_depth": 6, "n_estimators": 1000, "reg_alpha": 12.0}),
    "d6 gamma1 n1000 +min5TE": (M5, T_FIN, True, {"max_depth": 6, "n_estimators": 1000, "gamma": 1.0}),
    "d5 a12 n1500 +min5TE": (M5, T_FIN, True, {"max_depth": 5, "n_estimators": 1500, "reg_alpha": 12.0}),
}
_diag_models = {}
for _label, (_te, _tf, _ce, _pp) in CANDS.items():
    _A = build(train, _te, True, True, _tf, _ce)
    _B = build(evald, _te, True, True, _tf, _ce)
    _m = xgb.XGBClassifier(random_state=SEED, **{**BASE, **REF, "n_estimators": 3000, **_pp})
    _m.fit(_A, ytr)
    _diag_models[_label] = (_m, _B)
    print(f"diag {_label:26s} nfeat={_A.shape[1]:3d} eval_auc={roc_auc_score(yev, _m.predict_proba(_B)[:, 1]):.4f}")
_key = [k for k in ["d6 n1000 +min5TE", "d5 a12 n1500 +min5TE", "d6 a12 n1000 +min5TE"] if k in _diag_models]
if all(k in _diag_models for k in _key):
    _blend = np.mean([_diag_models[k][0].predict_proba(_diag_models[k][1])[:, 1] for k in _key], axis=0)
    print(f"diag blend top3        eval_auc={roc_auc_score(yev, _blend):.4f}")

# --- model: adopted configuration ----------------------------------------------
TIME_FEATURES, USE_TIME, USE_COUNTS, ACTIVE_TE, USE_CNT_EXTRA = set(T_FIN), True, True, M5, True
Xtr = prepare(train)
for n in M5:
    Xtr[f"te_{n}"] = oof_te[n]
model = xgb.XGBClassifier(random_state=SEED, **{**BASE, **REF, "max_depth": 6, "n_estimators": 1000})
t0 = time.time()
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
