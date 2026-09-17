"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Current best: bag of 5 diverse configs on calendar+MinOfDay+OrigHour features (0.7333).
This experiment: add DestHour / Origin-Dest hour features and re-bag; also long-route hour features.
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
_desthour_levels = pd.Index(sorted((train["Dest"].astype(str) + "_h" + _hour_tr.astype("Int64").astype(str)).unique()))
_carrhour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_h" + _hour_tr.astype("Int64").astype(str)).unique()))


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame, variant: str = "b1") -> pd.DataFrame:
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
    if variant in ("b2", "b3"):
        X["DestHour"] = pd.Categorical(df["Dest"].astype(str) + "_h" + hour.astype("Int64").astype(str),
                                       categories=_desthour_levels)
    if variant == "b3":
        X["CarrHour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_h" + hour.astype("Int64").astype(str),
                                       categories=_carrhour_levels)
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

CFGS = [dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8),
        dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=2.0, subsample=0.8),
        dict(n_estimators=600, max_depth=5, learning_rate=0.04, reg_lambda=5.0, alpha=4.0, subsample=0.8),
        dict(n_estimators=600, max_depth=6, learning_rate=0.04, reg_lambda=5.0, alpha=3.0, subsample=0.8),
        dict(n_estimators=450, max_depth=5, learning_rate=0.05, reg_lambda=5.0, alpha=3.0, subsample=0.8)]


def bag(variant):
    members = []
    for i, cfg in enumerate(CFGS):
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=100 + i,
                              n_jobs=N_JOBS, **cfg)
        m.fit(prepare(train, variant), ytr)
        members.append(m)
    return members


results = []
for variant in ("b1", "b2", "b3"):
    members = bag(variant)
    Xev = prepare(evald, variant)
    auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in members], axis=0))
    results.append((auc, variant, members))
    print(f"[probe] bag_{variant}: {auc:.4f}  ({time.time() - t0:.1f}s)")

best_auc, best_variant, best_members = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_variant}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df, best_variant)
    return np.mean([m.predict_proba(X)[:, 1] for m in best_members], axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
