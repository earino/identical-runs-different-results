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
CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
_dep_tr = train["DepTime"].astype(int)
HOUR_LEVELS = pd.Index(sorted((_dep_tr // 100).unique()))
HALFHOUR_LEVELS = pd.Index(sorted((_dep_tr // 30).unique()))
QUARTER_LEVELS = pd.Index(sorted((_dep_tr // 15).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    m = df["Month"].str.slice(2).astype(int)
    d = df["DayofMonth"].str.slice(2).astype(int)
    X["day_of_year"] = CUM_DAYS[m - 1] + d
    X["dow_n"] = df["DayOfWeek"].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    X["dep_hour"] = dep // 100
    X["dep_minutes_day"] = dep // 100 * 60 + dep % 100
    X["hour_cat"] = pd.Categorical(dep // 100, categories=HOUR_LEVELS)
    X["halfhour_cat"] = pd.Categorical(dep // 30, categories=HALFHOUR_LEVELS)
    X["quarter_cat"] = pd.Categorical(dep // 15, categories=QUARTER_LEVELS)
    X["log_distance"] = np.log1p(df["Distance"].astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
X_tr, y_tr = prepare(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)
models = []
for seed, cols, lam in [(42, 0.8, 2.0), (1, 0.8, 2.0), (7, 0.8, 2.0), (13, 0.8, 4.0), (99, 0.6, 2.0), (3, 0.6, 4.0), (5, 0.6, 4.0)]:
    m = xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=cols,
        min_child_weight=10,
        reg_lambda=lam,
        tree_method="hist",
        max_cat_to_onehot=100,
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=70,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s, best_iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
