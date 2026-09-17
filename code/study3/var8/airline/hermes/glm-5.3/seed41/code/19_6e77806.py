"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Best so far: 5-config bag on calendar+MinOfDay+OrigHour features (0.7333).
This experiment: chronological early stopping — hold out the last 15% of 2005 (by DayOfYear) as ES val,
train each member on the first 85%, pick n_rounds by ES, then REFIT? No: keep ES models directly.
Compare vs plain bag.
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
doy = Xtr["DayOfYear"].to_numpy()
cut = np.quantile(doy, 0.85)
val_mask = doy >= cut
print(f"[info] ES val rows: {val_mask.sum()} (doy>={cut:.0f})")
t0 = time.time()

CFGS = [
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=2.0, subsample=0.8),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=4.0, subsample=0.8),
    dict(n_estimators=600, max_depth=6, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8),
    dict(n_estimators=450, max_depth=5, learning_rate=0.05, reg_lambda=5.0, alpha=3.0, subsample=0.8),
]

Xfit, yfit = Xtr[~val_mask], ytr[~val_mask]
Xval, yval = Xtr[val_mask], ytr[val_mask]

variants = []

# control: plain bag (all data, fixed rounds)
members = []
for i, cfg in enumerate(CFGS):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=100 + i,
                         n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    members.append(m)
auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in members], axis=0))
variants.append((auc, "plain", members, None))
print(f"[probe] plain: {auc:.4f}  ({time.time() - t0:.1f}s)")

# chronological ES bag: cap n_estimators generously, ES rounds=50 on last-15%
es_members = []
es_iters = []
for i, cfg in enumerate(CFGS):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=100 + i,
                          n_jobs=N_JOBS, early_stopping_rounds=50, **dict(cfg, n_estimators=1200))
    m.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
    es_members.append(m)
    es_iters.append(m.best_iteration)
auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in es_members], axis=0))
variants.append((auc, "chronoes", es_members, es_iters))
print(f"[probe] chronoes: {auc:.4f} iters={es_iters}  ({time.time() - t0:.1f}s)")

best_auc, best_name, best_members, _ = max(variants, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    if best_name == "chronoes":
        # ES models expect the same feature set; predict without eval_set
        return np.mean([m.predict_proba(X)[:, 1] for m in best_members], axis=0)
    return np.mean([m.predict_proba(X)[:, 1] for m in best_members], axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
