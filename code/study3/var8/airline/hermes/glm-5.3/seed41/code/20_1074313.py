"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Best: 5-config bag on calendar+MinOfDay+OrigHour (0.7333).
This experiment: probe booster params (max_bin, gamma, mcw, eta/n) on ONE config, apply winner to the bag.
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


# --- setup -----------------------------------------------------------------------
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
t0 = time.time()

BAG = [
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=2.0, subsample=0.8),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=4.0, subsample=0.8),
    dict(n_estimators=600, max_depth=6, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8),
    dict(n_estimators=450, max_depth=5, learning_rate=0.05, reg_lambda=5.0, alpha=3.0, subsample=0.8),
]

# --- stage 1: single-config probes -----------------------------------------------
S = BAG[0]
probes = [
    ("ctrl", {}),
    ("maxbin512", dict(max_bin=512)),
    ("maxbin128", dict(max_bin=128)),
    ("gamma02", dict(gamma=0.2)),
    ("mcw2", dict(min_child_weight=2)),
    ("eta03_n800", dict(learning_rate=0.03, n_estimators=800)),
]
stage1 = []
for name, extra in probes:
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **dict(S, **extra))
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    stage1.append((auc, name, extra))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")

best_s1_auc, best_s1_name, best_extra = max(stage1, key=lambda r: r[0])
print(f"[probe] stage1 winner: {best_s1_name} ({best_s1_auc:.4f})")

# --- stage 2: apply winner to the bag, compare vs plain bag -----------------------
members_plain = []
for i, cfg in enumerate(BAG):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=100 + i,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    members_plain.append(m)
auc_plain = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in members_plain], axis=0))
print(f"[probe] bag_plain: {auc_plain:.4f}  ({time.time() - t0:.1f}s)")

members_new = []
for i, cfg in enumerate(BAG):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=100 + i,
                          n_jobs=N_JOBS, **dict(cfg, **best_extra))
    m.fit(Xtr, ytr)
    members_new.append(m)
auc_new = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in members_new], axis=0))
print(f"[probe] bag_{best_s1_name}: {auc_new:.4f}  ({time.time() - t0:.1f}s)")

if auc_new > auc_plain:
    final_members, final_auc = members_new, auc_new
else:
    final_members, final_auc = members_plain, auc_plain
print(f"[probe] final: {final_auc:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in final_members], axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
