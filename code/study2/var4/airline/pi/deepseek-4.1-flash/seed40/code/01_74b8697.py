"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design: an ensemble of heavily-regularized, shallow-ish XGBoost models over the raw feature set.
The eval split is a different year (2006 vs 2005) from training, so strong regularization plus
seed-averaging transfers much better than raw capacity (which overfits the training year).
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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# treat string columns as categoricals, but drop very high-cardinality ones
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


# --- model --------------------------------------------------------------------
# Two complementary strongly-regularized configurations (found by search); each is fit with several
# seeds and the probabilities are averaged. With colsample_bytree < 1 the seeds genuinely diversify.
CONFIGS = [
    dict(n_estimators=300, max_depth=6, learning_rate=0.05, colsample_bytree=0.8,
         min_child_weight=10, reg_lambda=20, reg_alpha=5, gamma=0.5),
    dict(n_estimators=800, max_depth=6, learning_rate=0.03, colsample_bytree=0.6,
         min_child_weight=10, reg_lambda=5, reg_alpha=0.5, gamma=0.5),
]
SEEDS = [1, 2, 3, 4, 5]

X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
for cfg in CONFIGS:
    for seed in SEEDS:
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
    preds = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return preds


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
