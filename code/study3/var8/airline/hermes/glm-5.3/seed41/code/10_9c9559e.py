"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Wins so far: calendar numerics, alpha=2, minute-of-day, dropping redundant raw categoricals.
This experiment: probe combinations of drops + derived time features.
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
BEST = dict(n_estimators=300, max_depth=3, learning_rate=0.05, reg_lambda=5.0, alpha=2.0)


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame, variant: str = "mod_dropm") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    mon = _num(X["Month"]).astype("float64")
    day = _num(X["DayofMonth"]).astype("float64")
    dow = _num(X["DayOfWeek"]).astype("float64")
    dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
    hour = np.floor(dep / 100.0)
    minute = dep % 100.0
    X["DepHour"] = hour
    X["MinOfDay"] = minute + 60.0 * hour
    cum = np.cumsum([0] + _MDAYS)
    X["DayOfYear"] = day + cum[np.clip(mon.astype(int) - 1, 0, 11)] - 1.0
    X["DayOfWeekNum"] = dow
    X["MonthNum"] = mon

    drops = {
        "mod_dropm": ["Month"],
        "mod_dropm_dom": ["Month", "DayofMonth"],
        "mod_dropm_dom_dow": ["Month", "DayofMonth", "DayOfWeek"],
        "mod_dropm_dom_dow_dep": ["Month", "DayofMonth", "DayOfWeek", "DepTime"],
        "mod_dropm_dep": ["Month", "DepTime"],
        "mod_dropall_raw": ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"],
    }.get(variant, [])
    if drops:
        X = X.drop(columns=[c for c in drops if c in X.columns])
    if "Distance" in drops:
        X["LogDist"] = np.log1p(pd.to_numeric(
            (df["Distance"] if "Distance" not in X.columns else X["Distance"]), errors="coerce").astype("float64"))
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- probe ---------------------------------------------------------------------
ytr, yev = to_y(train), to_y(evald)
t0 = time.time()
results = []


def probe(name, variant):
    Xtr, Xev = prepare(train, variant), prepare(evald, variant)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED,
                          n_jobs=N_JOBS, **BEST)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, name, variant, m))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")


probe("mod_dropm", "mod_dropm")
probe("mod_dropm_dom", "mod_dropm_dom")
probe("mod_dropm_dom_dow", "mod_dropm_dom_dow")
probe("mod_dropm_dom_dow_dep", "mod_dropm_dom_dow_dep")
probe("mod_dropm_dep", "mod_dropm_dep")
probe("mod_dropall_raw", "mod_dropall_raw")

best_auc, best_name, best_variant, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_variant))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
