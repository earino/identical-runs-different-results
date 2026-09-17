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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    mon = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    doy = (mon - 1) * 30.44 + dom
    X["doy"] = doy
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    for name, lo, hi in [
        ("newyear", 362, 366), ("newyear2", 1, 3), ("jul4", 181, 187),
        ("memorial", 145, 151), ("labor", 244, 250), ("thanks", 325, 331),
        ("xmas", 355, 361),
    ]:
        X["hol_" + name] = ((doy >= lo) & (doy <= hi)).astype(np.int8)

    # schedule-share features: within-input distribution of hour/dow/month per airport/carrier
    hh = (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int) // 100).clip(0, 23).astype(str)
    bases = {
        "Origin": df["Origin"].astype(str),
        "Dest": df["Dest"].astype(str),
        "UniqueCarrier": df["UniqueCarrier"].astype(str),
        "Route": df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
    }
    others = {"hour": hh, "dow": df["DayOfWeek"].astype(str), "month": df["Month"].astype(str)}
    for bname, base in bases.items():
        cb = np.maximum(base.map(base.value_counts()).to_numpy(dtype=float), 1.0)
        for oname, other in others.items():
            k = base + "_" + other
            ck = k.map(k.value_counts()).to_numpy(dtype=float)
            X[f"share_{bname}_{oname}"] = ck / cb
            X[f"share_{oname}_{bname}"] = ck / np.maximum(other.map(other.value_counts()).to_numpy(dtype=float), 1.0)
    # route composition within origin and dest
    ro = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy()
    co = np.maximum(df["Origin"].astype(str).map(df["Origin"].astype(str).value_counts()).to_numpy(dtype=float), 1.0)
    cd = np.maximum(df["Dest"].astype(str).map(df["Dest"].astype(str).value_counts()).to_numpy(dtype=float), 1.0)
    rk = pd.Series(ro)
    X["share_route_within_origin"] = rk.map(rk.value_counts()).to_numpy(dtype=float) / co
    X["share_route_within_dest"] = rk.map(rk.value_counts()).to_numpy(dtype=float) / cd
    for c in TE_COLS:
        X["te_" + c] = _te_key(df, c).map(TE_FULL[c]).fillna(PRIOR).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _hh_str(df):
    return (pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int) // 100).clip(0, 23).astype(str)


# --- out-of-fold target encodings for airport/carrier-hour delay propensity -----
y_tr = to_y(train)
PRIOR = float(y_tr.mean())
ALPHA = 50.0
TE_COLS = ["origin_hour", "dest_hour", "carrier_hour", "route"]


def _te_key(df, col):
    if col == "origin_hour":
        return df["Origin"].astype(str) + "_" + _hh_str(df)
    if col == "dest_hour":
        return df["Dest"].astype(str) + "_" + _hh_str(df)
    if col == "carrier_hour":
        return df["UniqueCarrier"].astype(str) + "_" + _hh_str(df)
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _fit_te(keys, y, prior):
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + prior * ALPHA) / (g["count"] + ALPHA)


TE_FULL = {c: _fit_te(_te_key(train, c), y_tr, PRIOR) for c in TE_COLS}
TE_OOF = {c: np.full(len(train), PRIOR) for c in TE_COLS}
_kf = KFold(5, shuffle=True, random_state=SEED)
for _tr, _va in _kf.split(train):
    _sub = train.iloc[_tr]
    _ys = y_tr[_tr]
    for _c in TE_COLS:
        _m = _fit_te(_te_key(_sub, _c), _ys, float(_ys.mean()))
        TE_OOF[_c][_va] = _te_key(train.iloc[_va], _c).map(_m).fillna(PRIOR).to_numpy()


# --- model --------------------------------------------------------------------
CONFIGS = [
    dict(max_depth=4, min_child_weight=10, n_estimators=300, colsample_bytree=0.8, reg_lambda=1.0),
    dict(max_depth=5, min_child_weight=20, n_estimators=320, colsample_bytree=0.7, reg_lambda=2.0),
    dict(max_depth=6, min_child_weight=40, n_estimators=350, colsample_bytree=0.6, reg_lambda=3.0),
    dict(max_depth=7, min_child_weight=60, n_estimators=300, colsample_bytree=0.6, reg_lambda=3.0),
    dict(max_depth=8, min_child_weight=80, n_estimators=260, colsample_bytree=0.6, reg_lambda=4.0),
]
SEEDS = [42, 7, 2024, 123, 555, 777]

models = []
Xtr = prepare(train)
for c in TE_COLS:
    Xtr["te_" + c] = TE_OOF[c]
ytr = to_y(train)
t0 = time.time()
for cfg in CONFIGS:
    for s in SEEDS:
        m = xgb.XGBClassifier(
            learning_rate=0.05,
            subsample=0.8,
            tree_method="hist",
            enable_categorical=True,
            random_state=s,
            n_jobs=N_JOBS,
            **cfg,
        )
        m.fit(Xtr, ytr)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
