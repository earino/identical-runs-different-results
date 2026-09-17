"""XGBoost binary classifier on the airline delay task.

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


def _to_int(series: pd.Series) -> pd.Series:
    """c-<n> string -> integer n."""
    return pd.to_numeric(series.astype(str).str.split("-").str[-1], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    # temporal numerics
    month = _to_int(df["Month"])
    dom = _to_int(df["DayofMonth"])
    dow = _to_int(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")

    X["month_n"] = month
    X["dom_n"] = dom
    X["dow_n"] = dow
    X["dep_hour"] = (dep // 100) % 24
    X["dep_minute"] = dep % 100
    X["dep_minofday"] = (dep % 1440) / 1440.0
    ang = 2 * np.pi * X["dep_minofday"]
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["dep_red_eye"] = ((dep >= 2000) | (dep < 600)).astype(float)
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    dw_ang = 2 * np.pi * (dow - 1) / 7.0
    X["dow_sin"] = np.sin(dw_ang)
    X["dow_cos"] = np.cos(dw_ang)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=200,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train),
    to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  (best_iter={model.best_iteration})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
