"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _cal(df: pd.DataFrame):
    """Month/DayofMonth/DayOfWeek arrive as c-<n> codes; recover the ordinals."""
    mon = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    return mon, dom, dow


# Holiday windows are the same every year, so they transfer to 2006 even though the level shifts.
_HOL = {(12, 15, 1, 5), (11, 18, 11, 30), (7, 1, 7, 7), (5, 25, 5, 31), (3, 25, 4, 5), (9, 1, 9, 5)}
# single busiest days (traffic peaks are calendar-fixed, so they carry over to 2006)
_PEAK = {(11, 27), (11, 28), (12, 23), (12, 26), (12, 27), (1, 2), (1, 3), (7, 3), (7, 5), (4, 1), (4, 3)}


def _day_of_year(mon, dom):
    return (mon - 1) * 31 + dom


def _holiday_proximity(doy):
    """0 inside a holiday window, else days to the nearest window edge (capped at 14)."""
    best = pd.Series(14.0, index=doy.index)
    for (m0, d0, m1, d1) in _HOL:
        a = _day_of_year(pd.Series([m0]), pd.Series([d0]))[0]
        b = _day_of_year(pd.Series([m1]), pd.Series([d1]))[0]
        if a <= b:
            dist = pd.Series(0.0, index=doy.index)
            dist = dist.mask(doy < a, a - doy).mask(doy > b, doy - b)
        else:  # window wraps the year end
            dist = pd.Series(0.0, index=doy.index)
            dist = dist.mask(doy > b, np.minimum(doy - b, 365 - a + doy + 1)).mask(doy < a, np.minimum(a - doy, 365 - a + doy + 1))
        best = np.minimum(best, dist)
    return best.clip(0, 14)


def _is_in_window(doy, m, d, m2, d2):
    a, b = _day_of_year(pd.Series([m]), pd.Series([d]))[0], _day_of_year(pd.Series([m2]), pd.Series([d2]))[0]
    if a <= b:
        return doy.between(a, b)
    return (doy >= a) | (doy <= b)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    t = pd.to_numeric(X["DepTime"], errors="coerce")
    valid = t.between(0, 2400) & (t % 100 < 60)
    mins = ((t // 100) * 60 + (t % 100)).where(valid)
    mins = mins.where(mins < 1440)
    X["dep_min"] = mins
    X["dep_hour"] = mins // 60
    X["dep_time_missing"] = t.isna().astype(np.int8)
    m = mins.fillna(720.0)
    X["dep_min_sin"] = np.sin(2 * np.pi * m / 1440.0)
    X["dep_min_cos"] = np.cos(2 * np.pi * m / 1440.0)
    X = X.drop(columns=["DepTime"])  # raw hhmm is non-monotone (2359 -> next-day 0000)
    mon, dom, dow = _cal(df)
    X["month_num"] = mon
    X["dom_num"] = dom
    X["dow_num"] = dow
    X["is_weekend"] = dow.isin([6, 7]).astype(np.int8)
    doy = _day_of_year(mon, dom)
    hol = pd.Series(False, index=X.index)
    for (m0, d0, m1, d1) in _HOL:
        hol |= _is_in_window(doy, m0, d0, m1, d1).fillna(False)
    X["is_holiday"] = hol.astype(np.int8)
    X["holiday_prox"] = _holiday_proximity(doy)
    X["is_peak_day"] = pd.Series(
        [(int(a), int(b)) in _PEAK for a, b in zip(mon.fillna(-1), dom.fillna(-1))], index=X.index
    ).astype(np.int8)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# 2005->2006 shift is large (inner 2005 holdout scores ~0.75, eval ~0.71), so the model must be forced
# onto structure that stays true across years. Departure delay grows monotonically with clock time;
# encoding that as a monotone constraint removes year-specific wiggles the tree would otherwise fit.
_MONO = {"dep_min": 1, "dep_hour": 1, "dep_min_sin": 0}
_mono_tuple = None


def _mono_constraints(cols):
    return tuple(_MONO.get(c, 0) for c in cols)


_BASE = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_lambda=50.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

# Bag of diverse regularized models: under the 2005->2006 shift, variance reduction is worth more than
# any single model's fit, and the diversity (depth / subsampling / seed) decorrelates the individual errors.
VARIANTS = [
    dict(seed=42, depth=6, mcw=50, sub=0.8, col=0.5, lam=50.0),
    dict(seed=7, depth=6, mcw=50, sub=0.7, col=0.4, lam=50.0),
    dict(seed=13, depth=6, mcw=80, sub=0.8, col=0.6, lam=50.0),
    dict(seed=2024, depth=5, mcw=50, sub=0.8, col=0.5, lam=50.0),
    dict(seed=99, depth=6, mcw=30, sub=0.7, col=0.5, lam=100.0),
    dict(seed=5, depth=7, mcw=50, sub=0.8, col=0.4, lam=50.0),
    dict(seed=77, depth=6, mcw=50, sub=0.9, col=0.5, lam=50.0),
    dict(seed=123, depth=5, mcw=80, sub=0.7, col=0.6, lam=50.0),
]

X_train = prepare(train)
y_train = to_y(train)
_mono = _mono_constraints(X_train.columns)

models = []
t0 = time.time()
for v in VARIANTS:
    p = dict(_BASE)
    p.update(random_state=v["seed"], max_depth=v["depth"], min_child_weight=v["mcw"],
             subsample=v["sub"], colsample_bytree=v["col"], reg_lambda=v["lam"],
             monotone_constraints=_mono)
    m = xgb.XGBClassifier(**p)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
