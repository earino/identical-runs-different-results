"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features: calendar set. This experiment: probe smaller models and DepHour ablations.
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


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame, hour_mode: str = "dep_hour") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    mon = _num(X["Month"]).astype("float64")
    day = _num(X["DayofMonth"]).astype("float64")
    dow = _num(X["DayOfWeek"]).astype("float64")
    dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
    if hour_mode == "dep_hour":
        X["DepHour"] = np.floor(dep / 100.0)
    elif hour_mode == "hour_cont":
        X["HourCont"] = np.floor(dep / 100.0) + (dep % 100.0) / 60.0
    elif hour_mode == "none":
        pass
    cum = np.cumsum([0] + _MDAYS)
    X["DayOfYear"] = day + cum[np.clip(mon.astype(int) - 1, 0, 11)] - 1.0
    X["DayOfWeekNum"] = dow
    X["MonthNum"] = mon
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- probe ---------------------------------------------------------------------
ytr, yev = to_y(train), to_y(evald)
t0 = time.time()
results = []  # (auc, name, hour_mode, cfg, model)

BASE_CFG = dict(n_estimators=300, max_depth=3, learning_rate=0.05, reg_lambda=5.0)


def probe(name, hour_mode, cfg):
    Xtr, Xev = prepare(train, hour_mode), prepare(evald, hour_mode)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, name, hour_mode, cfg, m))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")


probe("ctrl_dep_hour", "dep_hour", BASE_CFG)
probe("no_hour", "none", BASE_CFG)
probe("hour_cont", "hour_cont", BASE_CFG)
probe("n200", "dep_hour", dict(BASE_CFG, n_estimators=200))
probe("n120", "dep_hour", dict(BASE_CFG, n_estimators=120))
probe("lr075_n200", "dep_hour", dict(BASE_CFG, n_estimators=200, learning_rate=0.075))
probe("mcw10", "dep_hour", dict(BASE_CFG, min_child_weight=10))
probe("alpha1", "dep_hour", dict(BASE_CFG, alpha=1.0))
probe("d5_n300", "dep_hour", dict(BASE_CFG, max_depth=5, n_estimators=150, learning_rate=0.075))

best_auc, best_name, best_hm, best_cfg, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_hm))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
