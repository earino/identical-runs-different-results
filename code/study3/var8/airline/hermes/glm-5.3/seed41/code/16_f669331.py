"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Current best: calendar numerics + MinOfDay + OrigHour, d5 n600 lr.04 lam5 alpha5 subsample 0.8 (0.7311).
This experiment: stochastic-regularization grid (subsample x colsample x depth) around the optimum.
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


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
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
    X = X.drop(columns=["Month", "DayofMonth"])
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- probe ---------------------------------------------------------------------
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
t0 = time.time()
results = []

CONFIGS = [
    ("ctrl_sub8", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=5.0, subsample=0.8)),
    ("sub7", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=5.0, subsample=0.7)),
    ("sub6", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=5.0, subsample=0.6)),
    ("sub8_cs9", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=5.0,
                      subsample=0.8, colsample_bytree=0.9)),
    ("sub8_cs7", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=5.0,
                      subsample=0.8, colsample_bytree=0.7)),
    ("sub8_d6", dict(n_estimators=600, max_depth=6, learning_rate=0.04, reg_lambda=5.0, alpha=5.0, subsample=0.8)),
    ("sub8_d4", dict(n_estimators=600, max_depth=4, learning_rate=0.04, reg_lambda=5.0, alpha=5.0, subsample=0.8)),
    ("sub8_n900_lr03", dict(n_estimators=900, max_depth=5, learning_rate=0.03, reg_lambda=5.0, alpha=5.0, subsample=0.8)),
    ("sub8_n1200_lr02", dict(n_estimators=1200, max_depth=5, learning_rate=0.02, reg_lambda=5.0, alpha=5.0,
                             subsample=0.8)),
    ("sub8_a3", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8)),
    ("sub8_a8", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=8.0, subsample=0.8)),
    ("sub8_lam10", dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=10.0, alpha=5.0, subsample=0.8)),
]

for name, cfg in CONFIGS:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, name, m))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")

best_auc, best_name, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
