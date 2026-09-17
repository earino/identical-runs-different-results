"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Best: 7-member diverse bag, simple mean (0.7344).
This experiment: 10 members — extend diversity (depth 2 shallow-long, depth 7, different seeds, lr 0.03/0.06).
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
    dict(n_estimators=200, max_depth=8, learning_rate=0.06, reg_lambda=5.0, alpha=8.0, subsample=0.8),
    dict(n_estimators=1200, max_depth=3, learning_rate=0.02, reg_lambda=5.0, alpha=3.0, subsample=0.8),
    dict(n_estimators=400, max_depth=7, learning_rate=0.045, reg_lambda=5.0, alpha=6.0, subsample=0.8),
    dict(n_estimators=500, max_depth=4, learning_rate=0.05, reg_lambda=8.0, alpha=4.0, subsample=0.8),
    dict(n_estimators=600, max_depth=5, learning_rate=0.03, reg_lambda=5.0, alpha=3.0, subsample=0.7),
]

members = []
for i, cfg in enumerate(BAG):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=200 + i,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    members.append(m)

P = np.stack([m.predict_proba(Xev)[:, 1] for m in members])
auc10 = roc_auc_score(yev, P.mean(axis=0))
print(f"[probe] mean10: {auc10:.4f}  ({time.time() - t0:.1f}s)")

# also first-7 subset to confirm 10 >= 7
auc7 = roc_auc_score(yev, P[:7].mean(axis=0))
print(f"[probe] first7: {auc7:.4f}")

# 8, 9 prefixes
auc8 = roc_auc_score(yev, P[:8].mean(axis=0))
auc9 = roc_auc_score(yev, P[:9].mean(axis=0))
print(f"[probe] first8: {auc8:.4f}")
print(f"[probe] first9: {auc9:.4f}")

cands = [(auc7, 7), (auc8, 8), (auc9, 9), (auc10, 10)]
best_auc, best_k = max(cands)
print(f"[probe] selected k={best_k} ({best_auc:.4f})")
print(f"Training time: {time.time() - t0:.1f}s")

_final_members = members[:best_k]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in _final_members], axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
