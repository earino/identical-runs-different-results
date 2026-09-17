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

# congestion frequencies (fit on train only): share of scheduled departures per airport/hour, carrier/hour
_train_hour = (train["DepTime"] // 100).clip(0, 23)


def _congestion(df: pd.DataFrame, a: str, b) -> pd.Series:
    return df[a].astype(str) + "|" + pd.Series(b, index=df.index).astype(str)


CF = {
    "origin_hour": _congestion(train, "Origin", _train_hour).value_counts(normalize=True).to_dict(),
    "carrier_hour": _congestion(train, "UniqueCarrier", _train_hour).value_counts(normalize=True).to_dict(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hour = (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23)
    X["origin_hour_freq"] = _congestion(df, "Origin", hour).map(CF["origin_hour"]).astype(float)
    X["carrier_hour_freq"] = _congestion(df, "UniqueCarrier", hour).map(CF["carrier_hour"]).astype(float)
    # ordinal re-parameterisations of the c-<n> calendar columns: numeric order lets a single split express
    # seasonality / week position, which partition splits on the raw categoricals cannot
    cnum = lambda c: pd.to_numeric(df[c].astype(str).str.slice(2), errors="coerce")
    month, dom, dow = cnum("Month"), cnum("DayofMonth"), cnum("DayOfWeek")
    X["month_num"] = month
    X["dom_num"] = dom
    X["dow_num"] = dow
    X["day_of_year"] = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)[
        month.fillna(1).astype(int).clip(1, 12)
    ] + dom
    X["is_weekend"] = (dow >= 6).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: average of a hyperparameter-diverse bag of XGBoost fits --------------
# (seed, max_depth, min_child_weight, subsample, colsample_bytree, learning_rate)
MEMBERS = [
    (42, 5, 5, 0.9, 0.8, 0.03),
]


def _make_model(seed: int, depth: int, mcw: float, sub: float, cols: float, lr: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=depth,
        learning_rate=lr,
        min_child_weight=mcw,
        subsample=sub,
        colsample_bytree=cols,
        reg_lambda=5.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
        early_stopping_rounds=50,
        eval_metric="auc",
    )


t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = []
for cfg in MEMBERS:
    m = _make_model(*cfg)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  best_iterations={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
