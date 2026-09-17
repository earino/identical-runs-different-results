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
_tr_dep = pd.to_numeric(train["DepTime"], errors="coerce")
_tr_hour = (_tr_dep // 100).clip(0, 23).astype("Int64")
_tr_hh = (_tr_hour * 2 + (_tr_dep % 100 >= 30).astype("Int64")).astype(str)
cat_levels["carrier_hour"] = pd.Index(
    sorted((train["UniqueCarrier"].astype(str) + "_" + _tr_hh).unique())
)
_tr_db = pd.cut(pd.to_numeric(train["Distance"], errors="coerce"), bins=[0, 250, 500, 750, 1000, 1500, 1e9], labels=False).astype("Int64")
cat_levels["carrier_dist"] = pd.Index(
    sorted((train["UniqueCarrier"].astype(str) + "_" + _tr_db.astype(str)).unique())
)


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
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    _dep = pd.to_numeric(df["DepTime"], errors="coerce")
    _hour = (_dep // 100).clip(0, 23).astype("Int64")
    _hh = (_hour * 2 + (_dep % 100 >= 30).astype("Int64")).astype(str)
    X["carrier_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + _hh, categories=cat_levels["carrier_hour"]
    )
    _db = pd.cut(pd.to_numeric(df["Distance"], errors="coerce"), bins=[0, 250, 500, 750, 1000, 1500, 1e9], labels=False).astype("Int64")
    X["carrier_dist"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + _db.astype(str), categories=cat_levels["carrier_dist"]
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
BASE = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=16,
    n_jobs=N_JOBS,
)
SEEDS = [42, 7, 2024, 123, 555, 9, 77, 314, 271, 1618,
         11, 22, 33, 44, 55, 66, 88, 99, 111, 222]

Xtr = prepare(train)
ytr = to_y(train)

t0 = time.time()
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(random_state=s, **BASE)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
