"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]

# Columns dropped because their year-specific patterns do not transfer (train=2005, eval/holdout=2006).
DROP = ["Month", "DayofMonth", "Origin", "Dest", "UniqueCarrier"]

CAT_FEATURES = [c for c in ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"] if c not in DROP]
BASE_NUM = ["DepTime", "dep_hour", "dep_min", "is_weekend", "Distance"]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[RAW_COLS].copy()
    dep = pd.to_numeric(X["DepTime"], errors="coerce").fillna(0).astype(int)
    X["dep_hour"] = (dep // 100).clip(0, 23)
    X["dep_min"] = (dep % 100).clip(0, 59)
    X["is_weekend"] = X["DayOfWeek"].isin(["c-6", "c-7"]).astype(int)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _key(df: pd.DataFrame, cols) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).agg("|".join, axis=1)


_base_train = base_features(train)
cat_levels = {c: pd.Index(sorted(_base_train[c].dropna().unique())) for c in CAT_FEATURES}

# Smoothed target-encoding aggregates, fit on TRAINING data only. These capture interactions
# (route, airport/carrier x hour of day) that transfer across the 2005->2006 shift.
TE_SPECS = [
    ("te_origin", ["Origin"]),
    ("te_dest", ["Dest"]),
    ("te_carrier", ["UniqueCarrier"]),
    ("te_dow", ["DayOfWeek"]),
    ("te_route", ["Origin", "Dest"]),
    ("te_origin_hour", ["Origin", "dep_hour"]),
    ("te_dest_hour", ["Dest", "dep_hour"]),
    ("te_carrier_hour", ["UniqueCarrier", "dep_hour"]),
    ("te_dow_hour", ["DayOfWeek", "dep_hour"]),
    ("te_route_hour", ["Origin", "Dest", "dep_hour"]),
    ("te_carrier_origin_hour", ["UniqueCarrier", "Origin", "dep_hour"]),
    ("te_carrier_dest_hour", ["UniqueCarrier", "Dest", "dep_hour"]),
    ("te_origin_dow_hour", ["Origin", "DayOfWeek", "dep_hour"]),
    ("te_dest_dow_hour", ["Dest", "DayOfWeek", "dep_hour"]),
    ("te_carrier_route", ["UniqueCarrier", "Origin", "Dest"]),
    ("te_carrier_origin", ["UniqueCarrier", "Origin"]),
    ("te_carrier_dest", ["UniqueCarrier", "Dest"]),
    ("te_carrier_route_hour", ["UniqueCarrier", "Origin", "Dest", "dep_hour"]),
    ("te_carrier_dow_hour", ["UniqueCarrier", "DayOfWeek", "dep_hour"]),
    ("te_origin_dest_dow", ["Origin", "Dest", "DayOfWeek"]),
]
TE_SMOOTH = 40.0
TE_FEATURES = [n for n, _ in TE_SPECS]
feature_cols = BASE_NUM + CAT_FEATURES + TE_FEATURES

_prior = to_y(train).mean()
_te_maps = {}
for _name, _cols in TE_SPECS:
    _d = pd.DataFrame({"k": _key(_base_train, _cols).values, "y": to_y(train)})
    _g = _d.groupby("k")["y"].agg(["mean", "size"])
    _te_maps[_name] = (_g["mean"] * _g["size"] + _prior * TE_SMOOTH) / (_g["size"] + TE_SMOOTH)


def _te_frame(Xbase: pd.DataFrame, maps) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: _key(Xbase, cols).map(maps[name]).fillna(_prior).astype(float)
            for name, cols in TE_SPECS
        },
        index=Xbase.index,
    )


def _oof_te(Xbase: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Out-of-fold target encoding for the training frame (avoids leaking each row's own label)."""
    oof = {name: np.full(len(Xbase), _prior) for name in TE_FEATURES}
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(Xbase):
        for name, cols in TE_SPECS:
            _d = pd.DataFrame({"k": _key(Xbase.iloc[tr], cols).values, "y": y[tr]})
            _g = _d.groupby("k")["y"].agg(["mean", "size"])
            enc = (_g["mean"] * _g["size"] + _prior * TE_SMOOTH) / (_g["size"] + TE_SMOOTH)
            oof[name][va] = _key(Xbase.iloc[va], cols).map(enc).fillna(_prior).values
    return pd.DataFrame(oof, index=Xbase.index)


def prepare(df: pd.DataFrame, oof_y: np.ndarray = None) -> pd.DataFrame:
    """Feature path used by both training and predict_proba; statistics come from TRAIN only."""
    X = base_features(df)
    te = _te_frame(X, _te_maps) if oof_y is None else _oof_te(X, oof_y)
    X = pd.concat([X, te], axis=1)
    for c in CAT_FEATURES:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X[feature_cols]


# --- model --------------------------------------------------------------------
ytr = to_y(train)
Xtr = prepare(train, oof_y=ytr)

# Ensemble of XGBoost models: averaging decorrelated seeds/configs reduces variance and
# generalizes better across the distribution shift than any single model.
CONFIGS = [
    dict(max_depth=6, subsample=0.8, colsample_bytree=0.8, seed=42, n_estimators=400, learning_rate=0.05),
    dict(max_depth=6, subsample=0.8, colsample_bytree=0.8, seed=7, n_estimators=400, learning_rate=0.05),
    dict(max_depth=6, subsample=0.7, colsample_bytree=0.7, seed=2024, n_estimators=400, learning_rate=0.05),
    dict(max_depth=5, subsample=0.8, colsample_bytree=0.8, seed=11, n_estimators=400, learning_rate=0.05),
    dict(max_depth=7, subsample=0.8, colsample_bytree=0.8, seed=123, n_estimators=400, learning_rate=0.05),
    dict(max_depth=6, subsample=0.9, colsample_bytree=0.6, seed=99, n_estimators=400, learning_rate=0.05),
    dict(max_depth=5, subsample=0.7, colsample_bytree=0.6, seed=2025, n_estimators=600, learning_rate=0.03),
    dict(max_depth=7, subsample=0.7, colsample_bytree=0.7, seed=314, n_estimators=400, learning_rate=0.05),
    dict(max_depth=8, subsample=0.8, colsample_bytree=0.5, seed=271, n_estimators=400, learning_rate=0.05),
    dict(max_depth=6, subsample=0.6, colsample_bytree=0.9, seed=161, n_estimators=500, learning_rate=0.04),
    dict(max_depth=6, subsample=0.85, colsample_bytree=0.7, seed=808, n_estimators=500, learning_rate=0.04),
    dict(max_depth=4, subsample=0.9, colsample_bytree=0.8, seed=555, n_estimators=700, learning_rate=0.04),
]

t0 = time.time()
models = []
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
