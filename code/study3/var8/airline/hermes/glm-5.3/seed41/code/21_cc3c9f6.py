"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Best: 5-config bag on calendar+MinOfDay+OrigHour (0.7333).
This experiment: same, but try to squeeze the bag by adding two diverse deep members with high alpha,
and a rank-mean blend. Then keep whichever is best on eval.
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
]

members = []
for i, cfg in enumerate(BAG):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=100 + i,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr, ytr)
    members.append(m)

P = np.stack([m.predict_proba(Xev)[:, 1] for m in members])  # (7, n_eval)
auc_mean7 = roc_auc_score(yev, P.mean(axis=0))
print(f"[probe] mean7: {auc_mean7:.4f}  ({time.time() - t0:.1f}s)")

# weights: first 5 heavier (proven configs) vs uniform — try both plus rank-mean
w_uni = np.ones(7) / 7
w_heavy = np.array([1, 1, 1, 1, 1, 0.5, 0.5]); w_heavy = w_heavy / w_heavy.sum()
auc_h = roc_auc_score(yev, (P * w_heavy[:, None]).sum(axis=0))
R = np.stack([pd.Series(p).rank().to_numpy() for p in P])
auc_rank = roc_auc_score(yev, R.mean(axis=0))
print(f"[probe] weighted7: {auc_h:.4f}")
print(f"[probe] rank7: {auc_rank:.4f}  ({time.time() - t0:.1f}s)")

cands = [("mean7", auc_mean7, "mean"), ("weighted7", auc_h, "weighted"), ("rank7", auc_rank, "rank")]
best_auc, best_name, best_mode = max(cands)
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    Pd = np.stack([m.predict_proba(X)[:, 1] for m in members])
    if best_mode == "mean":
        return Pd.mean(axis=0)
    if best_mode == "weighted":
        return (Pd * w_heavy[:, None]).sum(axis=0)
    Rk = np.stack([pd.Series(p).rank().to_numpy() for p in Pd])
    # map ranks back to [0,1]
    return Rk.mean(axis=0) / (Pd.shape[1] + 1.0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
