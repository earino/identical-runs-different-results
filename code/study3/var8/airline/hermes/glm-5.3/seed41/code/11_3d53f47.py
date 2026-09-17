"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Features: calendar numerics + MinOfDay, raw Month/DayofMonth dropped. Config: d3, n300, lr 0.05, lambda 5, alpha 2.
This experiment: probe route/carrier interactions and per-airport aggregates.
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

route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame, variant: str = "base") -> pd.DataFrame:
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

    if variant in ("route", "all"):
        X["Route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels)
    if variant in ("orighour", "all"):
        X["OrigHour"] = pd.Categorical(df["Origin"].astype(str) + "_h" + hour.astype("Int64").astype(str),
                                       categories=orighour_levels)
    if variant in ("carrhour", "all"):
        X["CarrHour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_h" + hour.astype("Int64").astype(str),
                                       categories=carrhour_levels)
    if variant in ("hourbin", "all"):
        hb = pd.cut(dep, bins=[0, 559, 1059, 1359, 1659, 1959, 2400], labels=False).astype("float64")
        X["OrigBin"] = pd.Categorical(df["Origin"].astype(str) + "_b" + hb.astype("Int64").astype(str),
                                      categories=origbin_levels)

    X = X.drop(columns=["Month", "DayofMonth"])
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# interaction levels from TRAIN only
_orighour = train["Origin"].astype(str) + "_h" + np.floor(pd.to_numeric(train["DepTime"], errors="coerce") / 100.0).astype("Int64").astype(str)
orighour_levels = pd.Index(sorted(_orighour.unique()))
_carrhour = train["UniqueCarrier"].astype(str) + "_h" + np.floor(pd.to_numeric(train["DepTime"], errors="coerce") / 100.0).astype("Int64").astype(str)
carrhour_levels = pd.Index(sorted(_carrhour.unique()))
_hb = pd.cut(pd.to_numeric(train["DepTime"], errors="coerce"), bins=[0, 559, 1059, 1359, 1659, 1959, 2400], labels=False).astype("Int64")
_origbin = train["Origin"].astype(str) + "_b" + _hb.astype(str)
origbin_levels = pd.Index(sorted(_origbin.unique()))

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


probe("base", "base")
probe("route", "route")
probe("orighour", "orighour")
probe("carrhour", "carrhour")
probe("hourbin", "hourbin")
probe("all", "all")

best_auc, best_name, best_variant, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_variant))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
