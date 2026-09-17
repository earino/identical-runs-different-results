"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Current best: calendar numerics + MinOfDay + OrigHour, d5 n600 lr.04 lam5 alpha3 sub0.8 (0.7321).
This experiment: seed-bagged ensemble of the best config + a couple of nearby configs to check robustness.
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


def fit(cfg, seed):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=seed,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    return m


B = dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8)

variants = []

# single control
m0 = fit(B, SEED)
variants.append(("single", [m0], None))
print(f"[probe] single: {roc_auc_score(yev, m0.predict_proba(Xev)[:, 1]):.4f}  ({time.time() - t0:.1f}s)")

# seed bag of 5 (same config)
ms = [fit(B, s) for s in (11, 22, 33, 44, 55)]
variants.append(("bag5_same", ms, None))
print(f"[probe] bag5_same: {roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in ms], axis=0)):.4f}"
      f"  ({time.time() - t0:.1f}s)")

# diverse bag: 5 nearby configs
cfgs = [B,
        dict(B, alpha=2.0),
        dict(B, alpha=4.0),
        dict(B, max_depth=6),
        dict(B, n_estimators=450, learning_rate=0.05)]
ms2 = []
for i, c in enumerate(cfgs):
    ms2.append(fit(c, 100 + i))
variants.append(("bag5_diverse", ms2, cfgs))
print(f"[probe] bag5_diverse: {roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in ms2], axis=0)):.4f}"
      f"  ({time.time() - t0:.1f}s)")

# rank-average of diverse bag (rank-average handles calibration differences)
P = np.stack([m.predict_proba(Xev)[:, 1] for m in ms2])
R = np.stack([pd.Series(p).rank().to_numpy() for p in P])
variants.append(("bag5_diverse_rank", ms2, cfgs))
print(f"[probe] bag5_diverse_rank: {roc_auc_score(yev, R.mean(axis=0)):.4f}  ({time.time() - t0:.1f}s)")

best_name, best_members = None, None
best_auc = -1.0
for name, members, cfgs_ in variants:
    auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in members], axis=0))
    if auc > best_auc:
        best_auc, best_name, best_members = auc, name, members
print(f"[probe] selected {best_name}  auc={best_auc:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in best_members], axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
