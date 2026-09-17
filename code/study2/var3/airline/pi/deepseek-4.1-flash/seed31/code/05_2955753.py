"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# Month and DayofMonth encode 2005-specific seasonality that does not transfer to the 2006
# evaluation slice; dropping them gave a large cross-year AUC gain.
DROP_COLS = ["Month", "DayofMonth"]
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET] + DROP_COLS]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _dep_hour(df: pd.DataFrame) -> pd.Series:
    # DepTime is hhmm (e.g. 1357 = 13:57).
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    return (dep // 100) % 24


def _origin_hour(df: pd.DataFrame) -> pd.Series:
    # Airport-specific time-of-day congestion; stable structure across years.
    return df["Origin"].astype(str) + "_" + _dep_hour(df).astype(str)


# vocabulary fitted on training data only (engineered categorical)
cat_levels["origin_hour"] = pd.Index(sorted(_origin_hour(train).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    X["origin_hour"] = _origin_hour(df)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    X["minute_of_day"] = ((dep // 100) % 24) * 60 + (dep % 100)
    X["dep_minute_of_hour"] = dep % 100
    for c in cat_levels:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=20,
    learning_rate=0.03,
    colsample_bytree=0.7,
    reg_alpha=2.0,
    max_cat_threshold=128,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
