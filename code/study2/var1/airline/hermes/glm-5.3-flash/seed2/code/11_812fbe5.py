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


def _to_int(series):
    """c-<n> string -> integer n."""
    return pd.to_numeric(series.astype(str).str.split("-").str[-1], errors="coerce")


def _hour_col(series_like):
    """integer hour 0-23 from a pandas Series/Index of strings."""
    return (pd.to_numeric(pd.Series(series_like), errors="coerce") // 100 % 24)


# carrier x hour levels (fit on train only)
_carrier_hour_levels = pd.Index(sorted(
    (train["UniqueCarrier"].astype(str) + "_" + _hour_col(train["DepTime"]).astype("Int64").astype(str)).unique()
))


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
    # true minutes-of-day (handles hhmm >= 2400 by wrapping the hour)
    mod = ((dep // 100) % 24) * 60 + (dep % 100)
    X["dep_minofday"] = mod / 1440.0
    ang = 2 * np.pi * X["dep_minofday"]
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["dep_red_eye"] = ((dep >= 2000) | (dep < 600)).astype(float)
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    # hour as a nominal category (irregular delay profile by hour)
    X["dep_hour_cat"] = pd.Categorical(
        X["dep_hour"].astype("Int64").astype(str), categories=[str(h) for h in range(24)]
    )
    # carrier x hour interaction (480 levels)
    ch = (df["UniqueCarrier"].astype(str) + "_" + X["dep_hour"].astype("Int64").astype(str))
    X["carrier_hour"] = pd.Categorical(ch, categories=_carrier_hour_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
SEEDS = [42, 7, 2026, 123, 555]
DEPTHS = [8, 8, 6, 10, 8]


def make_model(seed: int, depth: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=depth,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bynode=0.8,
        reg_lambda=1.0,
        max_cat_to_onehot=1,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=200,
        eval_metric="auc",
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)
models = []
for i, seed in enumerate(SEEDS):
    m = make_model(seed, DEPTHS[i])
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"  seed {seed} depth {DEPTHS[i]}: best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    p = np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
