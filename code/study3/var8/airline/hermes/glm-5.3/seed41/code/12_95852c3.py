"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Best: calendar numerics + MinOfDay + Origin_x_hour categorical. This experiment: ORIG-hour neighborhood variants
(OrigHour+route, hour coarse/finer, airport-pair-hour), keeping feature count lean.
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

_dep_tr = pd.to_numeric(train["DepTime"], errors="coerce").astype("float64")
_hour_tr = np.floor(_dep_tr / 100.0)
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_orighour_levels = pd.Index(sorted((train["Origin"].astype(str) + "_h" + _hour_tr.astype("Int64").astype(str)).unique()))
_routecarr_levels = pd.Index(sorted((_route_tr + "_" + train["UniqueCarrier"].astype(str)).unique()))
# 2-hour bins
_h2_tr = (_hour_tr // 2).astype("Int64")
_orig2_levels = pd.Index(sorted((train["Origin"].astype(str) + "_h2_" + _h2_tr.astype(str)).unique()))
# route + hour coarse
_routecarr_orig_levels = pd.Index(sorted((_route_tr + "_" + train["Origin"].astype(str)).unique()))


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
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["DepHour"] = hour
    X["MinOfDay"] = (dep % 100.0) + 60.0 * hour
    cum = np.cumsum([0] + _MDAYS)
    X["DayOfYear"] = day + cum[np.clip(mon.astype(int) - 1, 0, 11)] - 1.0
    X["DayOfWeekNum"] = dow
    X["MonthNum"] = mon
    X["OrigHour"] = pd.Categorical(df["Origin"].astype(str) + "_h" + hour.astype("Int64").astype(str),
                                   categories=_orighour_levels)

    if variant in ("plus_routecarr", "plus2h", "plus_dowhour", "all"):
        X["RouteCarr"] = pd.Categorical(route + "_" + df["UniqueCarrier"].astype(str), categories=_routecarr_levels)
    if variant in ("plus2h", "all"):
        X["Orig2h"] = pd.Categorical(df["Origin"].astype(str) + "_h2_" + (hour // 2).astype("Int64").astype(str),
                                     categories=_orig2_levels)
    if variant in ("plus_dowhour", "all"):
        X["OrigDowHour"] = pd.Categorical(
            df["Origin"].astype(str) + "_d" + dow.astype("Int64").astype(str) + "_h" + hour.astype("Int64").astype(str),
            categories=_origdowhour_levels)

    X = X.drop(columns=["Month", "DayofMonth"])
    for c in cat_cols:
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_origdowhour_levels = pd.Index(sorted((train["Origin"].astype(str) + "_d"
                                       + _num(train["DayOfWeek"]).astype("Int64").astype(str)
                                       + "_h" + _hour_tr.astype("Int64").astype(str)).unique()))

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


probe("ctrl", "base")
probe("plus_routecarr", "plus_routecarr")
probe("plus2h", "plus2h")
probe("plus_dowhour", "plus_dowhour")
probe("all", "all")

best_auc, best_name, best_variant, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_variant))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
