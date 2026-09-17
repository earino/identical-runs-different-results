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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target encoding of stable interactions (fit on training data only) -------
# (airport/carrier) x (hour-of-day) delay propensities transfer across years, unlike raw Origin/Dest.
_HBINS = [0, 250, 500, 750, 1000, 1500, 2000, 6000]


def _hour(d):
    return ((d["DepTime"].astype(int) // 100) % 24).astype(str)


def _distb(d):
    return pd.cut(d["Distance"], _HBINS, labels=False).astype("Int64").astype(str)


def _minb(d):
    return pd.cut(d["DepTime"].astype(int) % 100, [-1, 15, 30, 45, 60], labels=False).astype("Int64").astype(str)


TE_KEYS = {
    "origin_hour": lambda d: d["Origin"].astype(str) + "_" + _hour(d),
    "dest_hour": lambda d: d["Dest"].astype(str) + "_" + _hour(d),
    "carrier_hour": lambda d: d["UniqueCarrier"].astype(str) + "_" + _hour(d),
    "dist_hour": lambda d: _distb(d) + "_" + _hour(d),
    "carrier_dow": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["DayOfWeek"].astype(str),
    "carrier_origin_hour": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Origin"].astype(str) + "_" + _hour(d),
    "carrier_dest_hour": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Dest"].astype(str) + "_" + _hour(d),
    "carrier_db_hour": lambda d: d["UniqueCarrier"].astype(str) + "_" + _distb(d) + "_" + _hour(d),
    "dest_db_hour": lambda d: d["Dest"].astype(str) + "_" + _distb(d) + "_" + _hour(d),
    "origin_db_hour": lambda d: d["Origin"].astype(str) + "_" + _distb(d) + "_" + _hour(d),
    "carrier_origin_db_hour": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Origin"].astype(str) + "_" + _distb(d) + "_" + _hour(d),
    "carrier_dest_db_hour": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Dest"].astype(str) + "_" + _distb(d) + "_" + _hour(d),
    "origin_h_mb": lambda d: d["Origin"].astype(str) + "_" + _hour(d) + "_" + _minb(d),
    "dest_h_mb": lambda d: d["Dest"].astype(str) + "_" + _hour(d) + "_" + _minb(d),
    "carrier_h_mb": lambda d: d["UniqueCarrier"].astype(str) + "_" + _hour(d) + "_" + _minb(d),
    "carrier_h_dow": lambda d: d["UniqueCarrier"].astype(str) + "_" + _hour(d) + "_" + d["DayOfWeek"].astype(str),
    "db_h_mb": lambda d: _distb(d) + "_" + _hour(d) + "_" + _minb(d),
    "carrier_h_mb_db": lambda d: d["UniqueCarrier"].astype(str) + "_" + _hour(d) + "_" + _minb(d) + "_" + _distb(d),
    "carrier_dest_h_mb": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Dest"].astype(str) + "_" + _hour(d) + "_" + _minb(d),
    "carrier_origin_h_mb": lambda d: d["UniqueCarrier"].astype(str) + "_" + d["Origin"].astype(str) + "_" + _hour(d) + "_" + _minb(d),
    "origin_h_mb_db": lambda d: d["Origin"].astype(str) + "_" + _hour(d) + "_" + _minb(d) + "_" + _distb(d),
    "dest_h_mb_db": lambda d: d["Dest"].astype(str) + "_" + _hour(d) + "_" + _minb(d) + "_" + _distb(d),
    "carrier_h_mb_dow": lambda d: d["UniqueCarrier"].astype(str) + "_" + _hour(d) + "_" + _minb(d) + "_" + d["DayOfWeek"].astype(str),
}
SMOOTH = 20.0
y_train = to_y(train)
PRIOR = float(y_train.mean())


def _smoothed(key: pd.Series, yy: np.ndarray) -> pd.Series:
    g = pd.DataFrame({"k": key.to_numpy(), "y": yy}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)


te_maps = {name: _smoothed(fn(train), y_train) for name, fn in TE_KEYS.items()}

# out-of-fold values for training rows so each row's own label does not leak in
te_oof = {name: np.full(len(train), PRIOR) for name in TE_KEYS}
from sklearn.model_selection import KFold

for tr, va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
    for name, fn in TE_KEYS.items():
        m = _smoothed(fn(train.iloc[tr]), y_train[tr])
        te_oof[name][va] = fn(train.iloc[va]).map(m).fillna(PRIOR).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for name, fn in TE_KEYS.items():
        X["te_" + name] = fn(df).map(te_maps[name]).fillna(PRIOR).astype(float)
    return X


# --- model ensemble (seed x config averaging) ---------------------------------
CONFIGS = [
    dict(n_estimators=250, max_depth=4, learning_rate=0.05, colsample_bytree=0.8, min_child_weight=5),
    dict(n_estimators=500, max_depth=3, learning_rate=0.05, colsample_bytree=0.8, min_child_weight=5),
    dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6, min_child_weight=10),
    dict(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5, min_child_weight=20),
    dict(n_estimators=350, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.4, min_child_weight=50, reg_lambda=5.0),
    dict(n_estimators=400, max_depth=10, learning_rate=0.02, subsample=0.8, colsample_bytree=0.3, min_child_weight=100, reg_lambda=10.0),
]
SEEDS = [42, 7]

Xtrain = prepare(train)
for name in TE_KEYS:
    Xtrain["te_" + name] = te_oof[name]
ytrain = to_y(train)
models = []
t0 = time.time()
for cfg in CONFIGS:
    for s in SEEDS:
        m = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            random_state=s,
            n_jobs=N_JOBS,
            **cfg,
        )
        m.fit(Xtrain, ytrain)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
