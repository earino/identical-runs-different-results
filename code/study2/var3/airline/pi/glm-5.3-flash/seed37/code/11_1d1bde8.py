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
SEED = 43

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
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_BAG = 200
BAG_FRAC = 0.8

_params = dict(
    n_estimators=50,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
_DEPTHS = [4, 5, 6, 7]
_LRS = [0.05, 0.1, 0.15]

t0 = time.time()
_X = prepare(train)
_y = to_y(train)
rng = np.random.RandomState(SEED)
_models = []
for k in range(N_BAG):
    m = xgb.XGBClassifier(random_state=SEED + k, max_depth=_DEPTHS[k % len(_DEPTHS)],
                          learning_rate=_LRS[k % len(_LRS)], **_params)
    idx = rng.choice(len(_X), size=int(BAG_FRAC * len(_X)), replace=False)
    m.fit(_X.iloc[idx], _y[idx])
    _models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    return ranks / (len(v) - 1.0)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([_ranks(m.predict_proba(Xp)[:, 1]) for m in _models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
