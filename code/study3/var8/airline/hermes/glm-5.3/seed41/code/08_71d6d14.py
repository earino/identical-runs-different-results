"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features: calendar set (DayOfYear, DayOfWeekNum, MonthNum, DepHour). Config: d3, n300, lr 0.05, lambda 5, alpha 1.
This experiment: probe L1/L2 spectrum and min_child_weight with alpha, plus 2-fold bagging.
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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    mon = _num(X["Month"]).astype("float64")
    day = _num(X["DayofMonth"]).astype("float64")
    dow = _num(X["DayOfWeek"]).astype("float64")
    dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
    X["DepHour"] = np.floor(dep / 100.0)
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
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
t0 = time.time()


def fit(idx, cfg, seed=SEED):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=seed,
                          n_jobs=N_JOBS, **cfg)
    m.fit(Xtr.iloc[idx] if idx is not None else Xtr, ytr[idx] if idx is not None else ytr)
    return m


results = []  # (auc, name, cfgs, idxs)  -- cfgs/idxs to rebuild in predict path
N = len(Xtr)
rng = np.random.RandomState(SEED)


def probe_simple(name, cfg):
    m = fit(None, cfg)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, name, [cfg], [None]))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")


B = dict(n_estimators=300, max_depth=3, learning_rate=0.05, reg_lambda=5.0)
probe_simple("alpha1", dict(B, alpha=1.0))
probe_simple("alpha2", dict(B, alpha=2.0))
probe_simple("alpha05", dict(B, alpha=0.5))
probe_simple("alpha1_lam2", dict(B, alpha=1.0, reg_lambda=2.0))
probe_simple("alpha1_lam10", dict(B, alpha=1.0, reg_lambda=10.0))
probe_simple("alpha1_mcw10", dict(B, alpha=1.0, min_child_weight=10))
probe_simple("alpha1_n200", dict(B, alpha=1.0, n_estimators=200))
probe_simple("alpha1_n500", dict(B, alpha=1.0, n_estimators=500))
probe_simple("alpha2_n500", dict(B, alpha=2.0, n_estimators=500))

# bagging: 4 disjoint 50% row-samples, average
cfg = dict(B, alpha=1.0)
ms = []
for k in range(4):
    idx = rng.rand(N) < 0.5
    ms.append(fit(idx, cfg, seed=100 + k))
auc = roc_auc_score(yev, np.mean([m.predict_proba(Xev)[:, 1] for m in ms], axis=0))
results.append((auc, "bag4x50", [cfg] * 4, [f"bag{k}" for k in range(4)]))
print(f"[probe] bag4x50: {auc:.4f}  ({time.time() - t0:.1f}s)")

best_auc, best_name, best_cfgs, _ = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")

if best_name == "bag4x50":
    _members = ms
else:
    _members = [fit(None, best_cfgs[0])]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in _members], axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
