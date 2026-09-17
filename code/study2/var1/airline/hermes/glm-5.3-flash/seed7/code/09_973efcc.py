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

# interaction categorical levels (from train only)
_hours = ((pd.to_numeric(train["DepTime"], errors="coerce") // 100).clip(0, 48) * 60
          + pd.to_numeric(train["DepTime"], errors="coerce") % 100) % 1440 // 60
_train_hours = _hours.astype(int).astype(str)
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))
car_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _train_hours).unique()))
org_hour_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + _train_hours).unique()))
dest_hour_levels = pd.Index(sorted((train["Dest"].astype(str) + "_" + _train_hours).unique()))
dow_hour_levels = pd.Index(sorted((train["DayOfWeek"].astype(str) + "_" + _train_hours).unique()))


def _deptime_feats(d: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_numeric(d["DepTime"], errors="coerce")
    hh = (t // 100).clip(0, 48)
    mm = t % 100
    mins = ((hh * 60 + mm) % 1440).fillna(0.0)
    ang = 2 * np.pi * mins / 1440.0
    return pd.DataFrame(
        {
            "dep_hour": (mins // 60).astype(float),
            "dep_min": mm.astype(float),
            "dep_sin": np.sin(ang),
            "dep_cos": np.cos(ang),
        },
        index=d.index,
    )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = _deptime_feats(df)
    X["dep_hour"] = dt["dep_hour"]
    X["dep_sin"] = dt["dep_sin"]
    X["dep_cos"] = dt["dep_cos"]
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce").fillna(0.0))
    # interaction categoricals (learned by the model itself, no target statistics)
    X["route"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels
    )
    X["car_hour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "_" + dt["dep_hour"].astype(int).astype(str),
        categories=car_hour_levels,
    )
    X["org_hour"] = pd.Categorical(
        df["Origin"].astype(str) + "_" + dt["dep_hour"].astype(int).astype(str),
        categories=org_hour_levels,
    )
    X["dest_hour"] = pd.Categorical(
        df["Dest"].astype(str) + "_" + dt["dep_hour"].astype(int).astype(str),
        categories=dest_hour_levels,
    )
    X["dow_hour"] = pd.Categorical(
        df["DayOfWeek"].astype(str) + "_" + dt["dep_hour"].astype(int).astype(str),
        categories=dow_hour_levels,
    )
    # day-of-month bucket (beginning/middle/end of month)
    dom = pd.to_numeric(
        df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce"
    )
    X["dom_bucket"] = pd.cut(
        dom, bins=[0, 10, 20, 31], labels=["mb", "mm", "me"]
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr = prepare(train)
y_tr = to_y(train)

# internal split for early stopping: random 80/20
n = len(X_tr)
rng = np.random.RandomState(SEED)
idx = rng.permutation(n)
cut = int(n * 0.8)
tr_idx, va_idx = idx[:cut], idx[cut:]

model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=12,
    min_child_weight=5,
    subsample=0.6,
    colsample_bytree=0.4,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    X_tr.iloc[tr_idx],
    y_tr[tr_idx],
    eval_set=[(X_tr.iloc[va_idx], y_tr[va_idx])],
    verbose=False,
)
best_it = int(model.best_iteration)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={best_it}")

# refit on all data with the tuned round count
model = xgb.XGBClassifier(
    n_estimators=best_it + 1,
    learning_rate=0.02,
    max_depth=12,
    min_child_weight=5,
    subsample=0.6,
    colsample_bytree=0.4,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(X_tr, y_tr)
print(f"Refit time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
