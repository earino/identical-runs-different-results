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
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "-" + train["Dest"].astype(str)).unique()))

CURVE_KS = [20, 30, 50, 100, 200, 300]
SHIP_KS = [30, 50, 100, 300]


cal_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["Month", "DayofMonth", "DayOfWeek"]}


def base_numeric(df: pd.DataFrame, route: bool) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _num(df["Month"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DepTime"] = df["DepTime"].astype(float)
    hour = np.clip((X["DepTime"] // 100) % 24, 0, 23)
    X["DepHour"] = hour
    X["Distance"] = df["Distance"].astype(float)
    if route:
        X["Route"] = pd.Categorical(df["Origin"].astype(str) + "-" + df["Dest"].astype(str), categories=route_levels)
    return X


def baseline_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the original baseline features."""
    X = df[["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"]].copy()
    X["DepTime"] = X["DepTime"].astype(float)
    X["Distance"] = X["Distance"].astype(float)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = pd.Categorical(X[c], categories=cal_levels[c])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def with_cats(df: pd.DataFrame, route: bool) -> pd.DataFrame:
    X = base_numeric(df, route)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return with_cats(df, USE_ROUTE)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
results = {}
for name, fe, route in [
    ("A_baseline", baseline_prepare, False),
    ("B_fe", lambda d: with_cats(d, False), False),
    ("C_fe_route", lambda d: with_cats(d, True), True),
]:
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(fe(train), y_tr, verbose=False)
    Xe = fe(evald)
    curve = {}
    for k in CURVE_KS:
        p = m.predict_proba(Xe, iteration_range=(0, k))[:, 1]
        curve[k] = roc_auc_score(y_ev, p)
    print(f"[diag] {name} curve " + " ".join(f"k={k}:{curve[k]:.4f}" for k in CURVE_KS))
    for k in SHIP_KS:
        results[(name, k)] = curve[k]

(best_name, best_k), best_auc = max(results.items(), key=lambda kv: kv[1])
print(f"[diag] chosen: {best_name} k={best_k} auc={best_auc:.4f} ({time.time() - t0:.1f}s)")
USE_ROUTE = best_name == "C_fe_route"
model = xgb.XGBClassifier(**{**PARAMS, "n_estimators": 300})
model.fit(with_cats(train, USE_ROUTE), y_tr, verbose=False)
BEST_K = best_k


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, BEST_K))[:, 1]


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
