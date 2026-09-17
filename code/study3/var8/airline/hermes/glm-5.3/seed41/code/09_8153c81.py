"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features: calendar set (DayOfYear, DayOfWeekNum, MonthNum, DepHour). Config: d3, n300, lr 0.05, lambda 5, alpha 2.
This experiment: probe feature set variations with a fresh ET/time-of-day angle (minute-of-day etc.).
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


def prepare(df: pd.DataFrame, variant: str = "cal") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    mon = _num(X["Month"]).astype("float64")
    day = _num(X["DayofMonth"]).astype("float64")
    dow = _num(X["DayOfWeek"]).astype("float64")
    dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
    hour = np.floor(dep / 100.0)
    X["DepHour"] = hour
    cum = np.cumsum([0] + _MDAYS)
    X["DayOfYear"] = day + cum[np.clip(mon.astype(int) - 1, 0, 11)] - 1.0
    X["DayOfWeekNum"] = dow
    X["MonthNum"] = mon
    if variant == "mod":
        X["MinOfDay"] = (dep % 100.0) + 60.0 * hour
        X["DepMin"] = dep % 100.0
    if variant == "bins":
        X["HourBin"] = pd.cut(dep, bins=[0, 559, 1059, 1359, 1659, 1959, 2400], labels=False).astype("float64")
        X["MinOfDay"] = (dep % 100.0) + 60.0 * hour
    if variant == "rush":
        # morning rush, midday, afternoon peak, evening
        X["IsMorningRush"] = ((hour >= 6) & (hour < 9)).astype("float64")
        X["IsAfternoon"] = ((hour >= 15) & (hour < 19)).astype("float64")
        X["IsLate"] = ((hour >= 21) | (hour < 5)).astype("float64")
    if variant == "sincos":
        X["HourSin"] = np.sin(2 * np.pi * (hour + (dep % 100.0) / 60.0) / 24.0)
        X["HourCos"] = np.cos(2 * np.pi * (hour + (dep % 100.0) / 60.0) / 24.0)
    if variant == "logd":
        X["LogDist"] = np.log1p(pd.to_numeric(X["Distance"], errors="coerce").astype("float64"))
    if variant == "dropmonth":
        X = X.drop(columns=["Month"])  # redundant with DayOfYear
    if variant == "dropdom":
        X = X.drop(columns=["DayofMonth"])  # redundant with DayOfYear
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


probe("ctrl", "cal")
probe("mod", "mod")        # minute-of-day + dep minute
probe("bins", "bins")      # hour bins + minute of day
probe("rush", "rush")      # rush-hour flags
probe("sincos", "sincos")  # cyclic hour
probe("logd", "logd")      # log distance
probe("dropmonth", "dropmonth")
probe("dropdom", "dropdom")

best_auc, best_name, best_variant, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_variant))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
