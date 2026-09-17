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

# --- static feature bookkeeping (fit on training data only) -------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _hours(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    t = np.where(t >= 2400, t - 2400, t)
    return pd.Series((t // 100).astype(int), index=df.index)


def _half(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    t = np.where(t >= 2400, t - 2400, t)
    return pd.Series(((t // 100) * 2 + ((t % 100) // 30)).astype(int), index=df.index)


def _quarter(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    t = np.where(t >= 2400, t - 2400, t)
    return pd.Series(((t // 100) * 4 + ((t % 100) // 15)).astype(int), index=df.index)


def _bin10(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    t = np.where(t >= 2400, t - 2400, t)
    return pd.Series(((t // 100) * 6 + ((t % 100) // 10)).astype(int), index=df.index)


def _exact(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    t = np.where(t >= 2400, t - 2400, t)
    return pd.Series((t // 5).astype(int), index=df.index)


def _mod(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0)
    t = np.where(t >= 2400, t - 2400, t)
    return pd.Series(((t // 100) * 60 + (t % 100)).astype(int), index=df.index)


def _cnt_key(df: pd.DataFrame, kind: str) -> pd.Series:
    O = df["Origin"].astype(str)
    D = df["Dest"].astype(str)
    C = df["UniqueCarrier"].astype(str)
    H = _hours(df).astype(str)
    HH = _half(df).astype(str)
    Q = _quarter(df).astype(str)
    B10 = _bin10(df).astype(str)
    EX = _exact(df).astype(str)
    if kind == "origin_hour":
        return O + "|" + H
    if kind == "dest_hour":
        return D + "|" + H
    if kind == "carrier_hour":
        return C + "|" + H
    if kind == "origin_dow":
        return O + "|" + df["DayOfWeek"].astype(str)
    if kind == "dest_dow":
        return D + "|" + df["DayOfWeek"].astype(str)
    if kind == "carrier_origin":
        return C + "|" + O
    if kind == "carrier_dest":
        return C + "|" + D
    if kind == "route":
        return O + "|" + D
    if kind == "route_hour":
        return O + "|" + D + "|" + H
    if kind == "origin_month":
        return O + "|" + df["Month"].astype(str)
    if kind == "dest_month":
        return D + "|" + df["Month"].astype(str)
    if kind == "origin_hour_dow":
        return O + "|" + H + "|" + df["DayOfWeek"].astype(str)
    if kind == "dest_hour_dow":
        return D + "|" + H + "|" + df["DayOfWeek"].astype(str)
    if kind == "carrier_origin_hour":
        return C + "|" + O + "|" + H
    if kind == "route_dow":
        return O + "|" + D + "|" + df["DayOfWeek"].astype(str)
    if kind == "origin_dom":
        return O + "|" + df["DayofMonth"].astype(str)
    if kind == "carrier_route":
        return C + "|" + O + "|" + D
    if kind == "dest_dom":
        return D + "|" + df["DayofMonth"].astype(str)
    if kind == "origin_month_hour":
        return O + "|" + df["Month"].astype(str) + "|" + H
    if kind == "carrier_dow":
        return C + "|" + df["DayOfWeek"].astype(str)
    if kind == "carrier_month":
        return C + "|" + df["Month"].astype(str)
    if kind == "route_month":
        return O + "|" + D + "|" + df["Month"].astype(str)
    if kind == "origin_half":
        return O + "|" + HH
    if kind == "dest_half":
        return D + "|" + HH
    if kind == "carrier_half":
        return C + "|" + HH
    if kind == "origin_quarter":
        return O + "|" + Q
    if kind == "dest_quarter":
        return D + "|" + Q
    if kind == "carrier_quarter":
        return C + "|" + Q
    if kind == "origin_half_dow":
        return O + "|" + HH + "|" + df["DayOfWeek"].astype(str)
    if kind == "route_half":
        return O + "|" + D + "|" + HH
    if kind == "origin_10":
        return O + "|" + B10
    if kind == "dest_10":
        return D + "|" + B10
    if kind == "carrier_10":
        return C + "|" + B10
    if kind == "origin_exact":
        return O + "|" + EX
    if kind == "route_quarter":
        return O + "|" + D + "|" + Q
    if kind == "origin_quarter_dow":
        return O + "|" + Q + "|" + df["DayOfWeek"].astype(str)
    if kind == "carrier_origin_half":
        return C + "|" + O + "|" + HH
    raise ValueError(kind)


CNT_KINDS = ["origin_hour", "dest_hour", "carrier_hour", "origin_dow", "dest_dow",
             "carrier_origin", "carrier_dest", "route", "route_hour", "origin_month", "dest_month",
             "origin_hour_dow", "dest_hour_dow", "carrier_origin_hour", "route_dow",
             "origin_dom", "carrier_route", "dest_dom", "origin_month_hour", "carrier_dow",
             "carrier_month", "route_month", "origin_half", "dest_half", "carrier_half",
             "origin_quarter", "dest_quarter", "carrier_quarter", "origin_half_dow", "route_half",
             "origin_10", "dest_10", "carrier_10", "origin_exact", "route_quarter",
             "origin_quarter_dow", "carrier_origin_half"]
cnt_maps = {k: _cnt_key(train, k).value_counts(normalize=True).to_dict() for k in CNT_KINDS}

_train_mod = _mod(train).to_numpy()


def _bank_keys(df: pd.DataFrame, name: str) -> np.ndarray:
    O = df["Origin"].astype(str)
    if name == "origin":
        return O.to_numpy()
    if name == "dest":
        return df["Dest"].astype(str).to_numpy()
    if name == "carrier":
        return df["UniqueCarrier"].astype(str).to_numpy()
    if name == "route":
        return (O + "|" + df["Dest"].astype(str)).to_numpy()
    if name == "origin_dow":
        return (O + "|" + df["DayOfWeek"].astype(str)).to_numpy()
    if name == "dest_dow":
        return (df["Dest"].astype(str) + "|" + df["DayOfWeek"].astype(str)).to_numpy()
    raise ValueError(name)


BANK_NAMES = ["origin", "dest", "carrier", "route", "origin_dow", "dest_dow"]
_bank_src = {n: {k: np.sort(v.to_numpy()) for k, v in pd.Series(_train_mod).groupby(_bank_keys(train, n))}
             for n in BANK_NAMES}


def _bank_count(df: pd.DataFrame, name: str, w: int) -> np.ndarray:
    keys = _bank_keys(df, name)
    mods = _mod(df).to_numpy()
    out = np.zeros(len(df), dtype=float)
    for key, idx in pd.Series(np.arange(len(df)), index=keys).groupby(level=0):
        arr = _bank_src[name].get(key)
        if arr is None:
            continue
        ii = idx.to_numpy()
        q = mods[ii]
        out[ii] = np.searchsorted(arr, q + w, "right") - np.searchsorted(arr, q - w, "left")
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hours = _hours(df)
    X["dep_hour_cat"] = pd.Categorical(hours, categories=list(range(24)))
    X["dep_minute"] = (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) % 100).astype(float)
    for k in CNT_KINDS:
        X[f"cnt_{k}"] = _cnt_key(df, k).map(cnt_maps[k]).fillna(0.0).astype(float)
    X["bank_origin_30"] = _bank_count(df, "origin", 30)
    X["bank_dest_30"] = _bank_count(df, "dest", 30)
    X["bank_carrier_30"] = _bank_count(df, "carrier", 30)
    X["bank_origin_60"] = _bank_count(df, "origin", 60)
    X["bank_origin_15"] = _bank_count(df, "origin", 15)
    X["bank_origin_90"] = _bank_count(df, "origin", 90)
    X["bank_dest_60"] = _bank_count(df, "dest", 60)
    X["bank_carrier_60"] = _bank_count(df, "carrier", 60)
    X["bank_route_30"] = _bank_count(df, "route", 30)
    X["bank_dest_15"] = _bank_count(df, "dest", 15)
    X["bank_dest_90"] = _bank_count(df, "dest", 90)
    X["bank_carrier_90"] = _bank_count(df, "carrier", 90)
    X["bank_origin_120"] = _bank_count(df, "origin", 120)
    X["bank_origin_dow_30"] = _bank_count(df, "origin_dow", 30)
    X["bank_dest_dow_30"] = _bank_count(df, "dest_dow", 30)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_all = to_y(train)
X_tr = prepare(train)

# --- model ensemble -----------------------------------------------------------
CONFIGS = []
for sd in (1, 2, 3, 4):
    CONFIGS += [
        dict(max_depth=3, n_estimators=200, learning_rate=0.1, seed=sd, colsample=0.8, subsample=0.8),
        dict(max_depth=4, n_estimators=120, learning_rate=0.1, seed=sd, colsample=0.7, subsample=0.85),
        dict(max_depth=5, n_estimators=80, learning_rate=0.08, seed=sd, colsample=0.7, subsample=0.85),
        dict(max_depth=6, n_estimators=60, learning_rate=0.08, seed=sd, colsample=0.7, subsample=0.85),
    ]
for sd in (1, 2):
    CONFIGS += [
        dict(max_depth=2, n_estimators=300, learning_rate=0.1, seed=sd, colsample=0.8, subsample=0.9),
        dict(max_depth=7, n_estimators=50, learning_rate=0.06, seed=sd, colsample=0.7, subsample=0.85),
    ]

models = []
t0 = time.time()
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample"],
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
