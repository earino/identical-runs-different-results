"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

DERIVED = ["hour", "minute", "tod", "origin_hour", "dest_hour", "carrier_hour"]


def _augment(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    dep = X["DepTime"]
    hour = dep // 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = hour * 60 + X["minute"]
    X["origin_hour"] = X["Origin"].astype(str) + "_" + hour.astype(str)
    X["dest_hour"] = X["Dest"].astype(str) + "_" + hour.astype(str)
    X["carrier_hour"] = X["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
    return X


# interaction levels come from TRAINING rows only
_tr_aug = _augment(train)
INTER_COLS = ["origin_hour", "dest_hour", "carrier_hour"]
INTER_LEVELS = {c: pd.Index(sorted(_tr_aug[c].unique())) for c in INTER_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _augment(df)
    X = X[feature_cols + DERIVED]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in INTER_COLS:
        X[c] = pd.Categorical(X[c], categories=INTER_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- 105-model bag, 50 trees per member ------------------------------------------
Xtr, ytr = prepare(train), to_y(train)
rng = np.random.RandomState(SEED)
n = len(Xtr)
models = []
t0 = time.time()
b = 0
CONFIGS = ((6, 1.0), (6, 0.8), (5, 0.8), (4, 0.7), (8, 0.8))
for depth, cs in CONFIGS:
    for _ in range(21):
        idx = rng.randint(0, n, n)
        m = xgb.XGBClassifier(
            n_estimators=50, max_depth=depth, learning_rate=0.1, tree_method="hist",
            colsample_bytree=cs, enable_categorical=True, random_state=SEED + b, n_jobs=N_JOBS,
        )
        m.fit(Xtr.iloc[idx], ytr[idx])
        models.append(m)
        b += 1
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
