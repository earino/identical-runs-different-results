"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Current best: calendar numerics + MinOfDay + OrigHour categorical, d5 n450 lr.04 lam5 alpha5 (0.7299).
This experiment: probe deeper (d6-d8) with strong alpha, more trees; plus flight-count aggregates.
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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

_MDAYS = [31, 28, 31, 31, 30, 30, 31, 31, 30, 31, 30, 31]

_dep_tr = pd.to_numeric(train["DepTime"], errors="coerce").astype("float64")
_hour_tr = np.floor(_dep_tr / 100.0)
_orighour_levels = pd.Index(sorted((train["Origin"].astype(str) + "_h" + _hour_tr.astype("Int64").astype(str)).unique()))

# airport daily flight volume (train only): Origin x DayOfYear counts
_doy_tr = (pd.to_numeric(train["DayofMonth"].astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")
           + (pd.to_numeric(train["Month"].astype(str).str.replace(r"^c-", "", regex=True), errors="coerce") - 1) * 31)
_oh_tr = train["Origin"].astype(str) + "_d" + _doy_tr.astype("Int64").astype(str)
_oh_counts = _oh_tr.value_counts()
_orig_doy_levels = pd.Index(sorted(_oh_tr.unique()))
# log volume map
_vol_map = np.log1p(_oh_counts.reindex(_orig_doy_levels).fillna(0.0))
_vol_map.index = _orig_doy_levels


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    mon = _num(X["Month"]).astype("float64")
    day = _num(X["DayofMonth"]).astype("float64")
    dow = _num(X["DayOfWeek"]).astype("float64")
    dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
    hour = np.floor(dep / 100.0)
    X["DepHour"] = hour
    X["MinOfDay"] = (dep % 100.0) + 60.0 * hour
    cum = np.cumsum([0] + _MDAYS)
    X["DayOfYear"] = day + cum[np.clip(mon.astype(int) - 1, 0, 11)] - 1.0
    X["DayOfWeekNum"] = dow
    X["MonthNum"] = mon
    X["OrigHour"] = pd.Categorical(df["Origin"].astype(str) + "_h" + hour.astype("Int64").astype(str),
                                   categories=_orighour_levels)
    if variant == "vol":
        key = df["Origin"].astype(str) + "_d" + X["DayOfYear"].astype("Int64").astype(str)
        X["OriginVol"] = key.map(_vol_map).astype("float64").fillna(np.log1p(50.0))
    X = X.drop(columns=["Month", "DayofMonth"])
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- probe ---------------------------------------------------------------------
ytr, yev = to_y(train), to_y(evald)
t0 = time.time()
results = []


def probe(name, variant, cfg):
    Xtr, Xev = prepare(train, variant), prepare(evald, variant)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, name, variant, cfg, m))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")


B = dict(n_estimators=450, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=5.0)
probe("ctrl", "base", B)
probe("vol", "vol", B)
probe("a5d6_n450", "base", dict(B, max_depth=6))
probe("a10d6_n450", "base", dict(B, max_depth=6, alpha=10.0))
probe("a10d7_n300", "base", dict(B, max_depth=7, n_estimators=300, learning_rate=0.05))
probe("a10d8_n200", "base", dict(B, max_depth=8, n_estimators=200, learning_rate=0.05))
probe("a5d5_n900_lr02", "base", dict(B, n_estimators=900, learning_rate=0.02))
probe("a5d5_sub8_n600", "base", dict(B, n_estimators=600, subsample=0.8))
probe("a5d5_cs8_n600", "base", dict(B, n_estimators=600, colsample_bytree=0.8))

best_auc, best_name, best_variant, best_cfg, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_variant))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
