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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

ytr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = ytr.mean()
SMOOTH = 20.0


def _dep_hour(df: pd.DataFrame) -> np.ndarray:
    return np.clip(df["DepTime"].to_numpy() // 100, 0, 23)


def _key(df: pd.DataFrame, name: str) -> pd.Series:
    h = pd.Series(_dep_hour(df), index=df.index).astype(str)
    if name == "route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    if name == "carrier":
        return df["UniqueCarrier"].astype(str)
    if name == "origin":
        return df["Origin"].astype(str)
    if name == "dest":
        return df["Dest"].astype(str)
    if name == "carrier_hour":
        return df["UniqueCarrier"].astype(str) + "_" + h
    if name == "origin_hour":
        return df["Origin"].astype(str) + "_" + h
    if name == "dest_hour":
        return df["Dest"].astype(str) + "_" + h
    if name == "route_hour":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + h
    if name == "carrier_origin":
        return df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    if name == "carrier_dest":
        return df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str)
    if name == "route_month":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + df["Month"].astype(str)
    raise KeyError(name)


TE_KEYS = ["route", "carrier", "origin", "dest", "carrier_hour", "origin_hour", "dest_hour", "route_hour", "route_month", "carrier_origin", "carrier_dest"]

# target-encoding statistics are fit on TRAINING data only; reused verbatim for unseen rows.
_tr_keys = {k: _key(train, k) for k in TE_KEYS}
te_maps = {}
cnt_maps = {}
for k in TE_KEYS:
    st = pd.DataFrame({"k": _tr_keys[k], "y": ytr}).groupby("k")["y"].agg(["sum", "count"])
    te_maps[k] = (st["sum"] + prior * SMOOTH) / (st["count"] + SMOOTH)
    cnt_maps[k] = st["count"]

# out-of-fold encodings for the training rows (avoids leakage)
oof_te = {k: np.full(len(train), prior) for k in TE_KEYS}
for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
    for k in TE_KEYS:
        key = _tr_keys[k]
        st = pd.DataFrame({"k": key.iloc[_tr], "y": ytr[_tr]}).groupby("k")["y"].agg(["sum", "count"])
        m = (st["sum"] + prior * SMOOTH) / (st["count"] + SMOOTH)
        oof_te[k][_va] = key.iloc[_va].map(m).fillna(prior).to_numpy()


def prepare(df: pd.DataFrame, _oof: bool = False) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for k in TE_KEYS:
        key = _key(df, k)
        if _oof:
            X["te_" + k] = oof_te[k]
        else:
            X["te_" + k] = key.map(te_maps[k]).fillna(prior).to_numpy()
        X["cnt_" + k] = key.map(cnt_maps[k]).fillna(0).to_numpy()
    X["frac_route_hour"] = (X["cnt_route_hour"] / X["cnt_route"].clip(lower=1)).to_numpy()
    X["frac_origin_hour"] = (X["cnt_origin_hour"] / X["cnt_origin"].clip(lower=1)).to_numpy()
    X["frac_dest_hour"] = (X["cnt_dest_hour"] / X["cnt_dest"].clip(lower=1)).to_numpy()
    _month = df["Month"].str.replace("c-", "", regex=False).astype(int)
    _dom = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["doy"] = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])[_month.to_numpy() - 1] + _dom.to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.4,
    min_child_weight=50,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 7, 123]

X_train = prepare(train, _oof=True)
t0 = time.time()
models = []
for _seed in SEEDS:
    _m = xgb.XGBClassifier(**BASE_PARAMS, random_state=_seed)
    _m.fit(X_train, ytr)
    models.append(_m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
