"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame, groups=frozenset({"time", "date", "dist", "route"})) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    if "time" in groups:
        dep = df["DepTime"]
        hour = dep // 100
        minute = dep % 100
        X["hour"] = hour
        X["minute"] = minute
        X["tod"] = hour * 60 + minute
    if "date" in groups:
        X["month_n"] = _cnum(df["Month"])
        X["day_n"] = _cnum(df["DayofMonth"])
        X["dow_n"] = _cnum(df["DayOfWeek"])
    if "dist" in groups:
        X["log_dist"] = np.log1p(df["Distance"])
    if "route" in groups:
        X["route"] = pd.Categorical(
            df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels
        )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
yall, yev = to_y(train), to_y(evald)

results = []
t0 = time.time()
for name, groups in [
    ("base", frozenset()),
    ("time", frozenset({"time"})),
    ("date", frozenset({"date"})),
    ("dist", frozenset({"dist"})),
    ("route", frozenset({"route"})),
    ("time+date", frozenset({"time", "date"})),
    ("time+date+dist", frozenset({"time", "date", "dist"})),
]:
    for depth, rounds in [(4, 100), (3, 200)]:
        m = xgb.XGBClassifier(
            n_estimators=rounds,
            max_depth=depth,
            learning_rate=0.1,
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        m.fit(prepare(train, groups), yall)
        auc = roc_auc_score(yev, m.predict_proba(prepare(evald, groups))[:, 1])
        results.append((auc, depth, rounds, groups, m))
        print(f"diag feat={name} depth={depth} rounds={rounds} auc={auc:.4f} t={time.time() - t0:.1f}s", flush=True)

best_auc, best_depth, best_rounds, best_groups, model = max(results, key=lambda r: r[0])
print(f"diag BEST groups={sorted(best_groups)} depth={best_depth} rounds={best_rounds} auc={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, best_groups))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
