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


def _day_of_year(mon, dom):
    return (mon - 1) * 31 + dom


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


model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

X_train = prepare(train)
y_train = to_y(train)
model.set_params(monotone_constraints=_mono_constraints(X_train.columns))

t0 = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
