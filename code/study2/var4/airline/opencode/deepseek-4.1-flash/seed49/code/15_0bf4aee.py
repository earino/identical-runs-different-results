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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype(int)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["dep_hour"] = hour.astype(np.int16)
    X["dep_hour_cat"] = pd.Categorical(hour, categories=list(range(24)))
    X["dep_quarter_cat"] = pd.Categorical((hour * 60 + minute) // 15, categories=list(range(96)))
    X["dep_minute"] = minute.astype(np.int16)
    X["dep_since_midnight"] = (hour * 60 + minute).astype(np.int16)
    month = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(1).astype(int)
    day = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(1).astype(int)
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    X["doy"] = (cum[(month - 1).clip(0, 11)] + day).astype(np.int16)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
from sklearn.model_selection import train_test_split

Xtr = prepare(train)
ytr = to_y(train)

BASE = dict(
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=1.0,
    max_cat_to_onehot=500,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

ENSEMBLE = list(zip((6, 7, 8, 9, 6, 7, 8, 9), (42, 7, 2024, 123, 55, 99, 17, 2025)))

t0 = time.time()
models = []
for depth, seed in ENSEMBLE:
    Xa, Xb, ya, yb = train_test_split(Xtr, ytr, test_size=0.1, random_state=seed, stratify=ytr)
    m = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=depth,
        early_stopping_rounds=50,
        random_state=seed,
        **BASE,
    )
    m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
    models.append(m)
    print(f"  d{depth} seed{seed} best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X), dtype=np.float64)
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
