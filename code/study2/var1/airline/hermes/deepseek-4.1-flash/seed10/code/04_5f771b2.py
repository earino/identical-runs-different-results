"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Model choice: train (2005) and eval (2006) are different years, so extra capacity overfits the year shift.
Shallow, slowly-learned boosters generalise best (depth 4, lr <= 0.05). The final predictor averages several
such boosters trained with different depths/learning rates/seeds, which lowers prediction variance.
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
# Shared regularisation; per-member depth / learning rate / seed.
MEMBERS = [
    dict(max_depth=4, learning_rate=0.025, n_estimators=400, seed=1),
    dict(max_depth=4, learning_rate=0.0125, n_estimators=800, seed=2),
    dict(max_depth=3, learning_rate=0.025, n_estimators=600, seed=3),
    dict(max_depth=5, learning_rate=0.02, n_estimators=300, seed=4),
    dict(max_depth=4, learning_rate=0.05, n_estimators=200, seed=5),
]

Xtr, ytr = prepare(train), to_y(train)
models = []
t0 = time.time()
for m in MEMBERS:
    md = dict(m)
    seed = md.pop("seed")
    est = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=N_JOBS,
        **md,
    )
    est.fit(Xtr, ytr)
    models.append(est)
    print(f"  member depth={md['max_depth']} lr={md['learning_rate']} n={md['n_estimators']} fit")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Train AUC: {roc_auc_score(ytr, predict_proba(train)):.4f}")


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
