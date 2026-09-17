"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives inside prepare(), and all fitted statistics are learned from the
training frame at import time, so predict_proba() reproduces the exact same transformation on unseen rows.
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
TWO_PI = 2.0 * np.pi

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature plan -------------------------------------------------------------
# Raw columns: Month, DayofMonth, DayOfWeek (c-<n>), DepTime (hhmm int), UniqueCarrier, Origin, Dest, Distance.
raw_cat = ["UniqueCarrier", "Origin", "Dest"]
feature_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime"] + raw_cat + ["Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in raw_cat}


def _to_int(s: pd.Series) -> pd.Series:
    """'c-12' -> 12, passthrough numeric."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _to_int(df["Month"])
    dom = _to_int(df["DayofMonth"])
    dow = _to_int(df["DayOfWeek"])
    dist = df["Distance"].astype(float).fillna(0.0)

    # time of day
    dt = df["DepTime"].astype(float).fillna(0.0)
    hour = (dt // 100).astype(float)
    minute = (dt % 100).astype(float)
    dep_min = hour * 60.0 + minute
    dep_min = dep_min.where(dep_min <= 1440, dep_min - 1440)  # fix rare hhmm typos > 2400

    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["is_weekend"] = (dow >= 6).astype(float)
    X["doy"] = month * 31.0 + dom
    X["dep_min"] = dep_min
    X["hour"] = dep_min / 60.0
    X["dep_sin"] = np.sin(TWO_PI * dep_min / 1440.0)
    X["dep_cos"] = np.cos(TWO_PI * dep_min / 1440.0)
    X["is_red_eye"] = ((dep_min < 360) | (dep_min >= 1260)).astype(float)
    X["dist"] = dist
    X["log_dist"] = np.log1p(dist)

    for c in raw_cat:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
params = dict(
    max_depth=7,
    learning_rate=0.07,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    reg_lambda=1.0,
)

# Hold out a random 15% of train for early stopping, then refit on all of train.
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = int(0.15 * len(train))
val_idx = np.sort(idx[:n_val])
fit_idx = np.sort(idx[n_val:])

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
es_model = xgb.XGBClassifier(n_estimators=500, early_stopping_rounds=30, **params)
es_model.fit(Xtr.iloc[fit_idx], ytr[fit_idx], eval_set=[(Xtr.iloc[val_idx], ytr[val_idx])], verbose=False)
best_iter = max(1, es_model.best_iteration + 1)
print(f"best_iteration={best_iter}  fit_time={time.time() - t0:.1f}s")

model = xgb.XGBClassifier(n_estimators=best_iter, **params)
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
