"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes (why these choices):
  * Eval is a different year (2006) from training (2005), so overfitting the training year is the main risk.
    We use an ensemble of two heavily-regularized XGBoost configs, each fit with several seeds and averaged.
  * Departure time-of-day is the strongest signal. Treating it as *categorical* buckets
    (24 hourly + 72 twenty-minute + 96 fifteen-minute buckets) beats the raw hhmm integer.
  * Month and DayofMonth are dropped: their delay patterns are year-specific and transfer poorly
    to the eval year, while DayOfWeek, carrier, airports and distance transfer well.
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# Columns whose year-specific effects do not transfer to the eval year.
DROP_COLS = ["Month", "DayofMonth"]


# --- feature engineering ------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """All derived features. Must be callable on unseen raw rows (predict_proba)."""
    X = df.copy()
    dt = pd.to_numeric(X["DepTime"], errors="coerce").fillna(0).astype(int)
    hour = (dt // 100) % 24
    minute = dt % 100
    minute = minute.where(minute < 60, 0)  # guard against malformed times (e.g. 2620)
    clock = hour * 60 + minute
    X["dep_hour"] = clock // 60
    X["dep_20"] = clock // 20
    X["dep_q"] = clock // 15
    X["logdist"] = np.log1p(X["Distance"])
    X["carhour"] = X["UniqueCarrier"].astype(str) + "_" + X["dep_hour"].astype(str)
    return X


train_fe = add_features(train)

feature_cols = [c for c in train_fe.columns if c not in ID_COLS + [TARGET] + DROP_COLS]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train_fe[c]) or pd.api.types.is_string_dtype(train_fe[c])]
cat_cols = [c for c in obj_cols if train_fe[c].nunique() <= 1000]
cat_cols += [c for c in ("dep_hour", "dep_20", "dep_q") if c in feature_cols]
# explicit level grids so unseen-but-valid times are handled consistently on the hidden holdout
cat_levels = {c: pd.Index(sorted(train_fe[c].dropna().unique())) for c in cat_cols}
cat_levels["dep_hour"] = pd.Index(range(24))
cat_levels["dep_20"] = pd.Index(range(72))
cat_levels["dep_q"] = pd.Index(range(96))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = add_features(df)[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# (params, seeds) pairs. Deep trees (depth 12-14) exploit the categorical time buckets best. n_estimators
# is kept moderate because deep-tree *prediction* is the wall-clock bottleneck under the 120s limit;
# averaging a few fast models beats one slow model. Total run ~80s.
MODEL_SPECS = [
    (dict(n_estimators=400, max_depth=14, learning_rate=0.04, colsample_bytree=0.5,
          min_child_weight=1, reg_lambda=1, reg_alpha=0.5, gamma=0.0), [1, 2, 3]),
    (dict(n_estimators=400, max_depth=14, learning_rate=0.04, colsample_bytree=0.7,
          min_child_weight=1, reg_lambda=5, reg_alpha=0.5, gamma=0.0), [1]),
]

X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
for cfg, seeds in MODEL_SPECS:
    for seed in seeds:
        m = xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            random_state=seed,
            n_jobs=N_JOBS,
            **cfg,
        )
        m.fit(X_train, y_train)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
