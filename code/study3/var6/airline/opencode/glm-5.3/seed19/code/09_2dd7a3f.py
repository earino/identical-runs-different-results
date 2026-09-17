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


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    # calendar / clock numerics
    for c, name in (("Month", "mon"), ("DayofMonth", "dom"), ("DayOfWeek", "dow")):
        X[name] = pd.to_numeric(X[c].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["blk"] = dep // 300  # 3-hour block of the day
    hf = X["hour"] + X["minute"] / 60.0
    X["sin_t"] = np.sin(2 * np.pi * hf / 24.0)
    X["cos_t"] = np.cos(2 * np.pi * hf / 24.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


# --- volume/congestion features: counts fit on TRAIN only ------------------------
_bf_tr = base_features(train)
_o = _bf_tr["Origin"].astype(str).to_numpy()
_d = _bf_tr["Dest"].astype(str).to_numpy()
_c = _bf_tr["UniqueCarrier"].astype(str).to_numpy()
_h = _bf_tr["blk"].astype("Int64").astype(str).to_numpy()
_dow = _bf_tr["dow"].astype("Int64").astype(str).to_numpy()

VOL_SPECS = {
    "vol_org_blk": (_o, _h),
    "vol_dest_blk": (_d, _h),
    "vol_org_dow": (_o, _dow),
    "vol_car_blk": (_c, _h),
}
vol_maps = {}
vol_defaults = {}
for k, (a, b) in VOL_SPECS.items():
    cnt = pd.Series(list(zip(a, b))).value_counts()
    vol_maps[k] = cnt
    vol_defaults[k] = float(cnt.mean())


def add_vol(X: pd.DataFrame) -> pd.DataFrame:
    o = X["Origin"].astype(str).to_numpy()
    d = X["Dest"].astype(str).to_numpy()
    c = X["UniqueCarrier"].astype(str).to_numpy()
    h = X["blk"].astype("Int64").astype(str).to_numpy()
    dow = X["dow"].astype("Int64").astype(str).to_numpy()
    keys = {"vol_org_blk": (o, h), "vol_dest_blk": (d, h), "vol_org_dow": (o, dow), "vol_car_blk": (c, h)}
    for k, (a, b) in keys.items():
        X[k] = np.log1p(pd.Series(list(zip(a, b))).map(vol_maps[k]).fillna(vol_defaults[k]).to_numpy())
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_vol(base_features(df))


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of XGBoost models ----------------------------------
MEMBERS = [
    dict(seed=42, leaves=48, mcw=10),
    dict(seed=7, leaves=64, mcw=10),
    dict(seed=123, leaves=96, mcw=20),
    dict(seed=2024, leaves=64, mcw=15),
    dict(seed=55, leaves=56, mcw=12),
]
t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
X_eval, y_eval = prepare(evald), to_y(evald)
models = []
for cfg in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=20,
        max_leaves=cfg["leaves"],
        grow_policy="lossguide",
        min_child_weight=cfg["mcw"],
        learning_rate=0.03,
        tree_method="hist",
        enable_categorical=True,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=200,
        eval_metric="auc",
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    models.append(m)
    print(f"  member seed={cfg['seed']} leaves={cfg['leaves']} best_iter={m.best_iteration} ({time.time() - t0:.1f}s)")
model = models  # ensemble container


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
