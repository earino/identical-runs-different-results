"""XGBoost binary classifier, improved: early stopping + refit on full data.

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
from sklearn.model_selection import train_test_split

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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=4000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=100,
)

# --- model --------------------------------------------------------------------
t0 = time.time()
X = prepare(train)
y = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)

pilot = xgb.XGBClassifier(**PARAMS)
pilot.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_n = int(pilot.best_iteration) + 1
print(f"Pilot best iteration: {best_n} (of {PARAMS['n_estimators']})")

model = xgb.XGBClassifier(**{**PARAMS, "n_estimators": best_n, "early_stopping_rounds": None})
model.fit(X, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
