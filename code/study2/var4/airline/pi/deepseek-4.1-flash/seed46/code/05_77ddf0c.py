"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
RAW_COLS = [c for c in RAW_COLS if c not in obj_cols or c in cat_cols]
TIME_COLS = ["dep_hour", "dep_min", "dep_minutes"]
feature_cols = RAW_COLS + TIME_COLS + ["origin_freq", "dest_freq", "carrier_freq"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
# frequency (airport/carrier size) encodings, fit on TRAIN ONLY
_freq = {c: train[c].value_counts() for c in ["Origin", "Dest", "UniqueCarrier"]}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[RAW_COLS].copy()
    dep = pd.to_numeric(X["DepTime"], errors="coerce")
    X["dep_hour"] = (dep // 100).clip(0, 23)
    X["dep_min"] = dep % 100
    X["dep_minutes"] = X["dep_hour"] * 60 + X["dep_min"]
    X["origin_freq"] = df["Origin"].map(_freq["Origin"]).fillna(0)
    X["dest_freq"] = df["Dest"].map(_freq["Dest"]).fillna(0)
    X["carrier_freq"] = df["UniqueCarrier"].map(_freq["UniqueCarrier"]).fillna(0)
    X = X[feature_cols]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# Diverse XGBoost members: different depths / feature subsampling / seeds. Each picks its own
# number of trees on an internal validation split (never eval.csv), then refits on all of train.
BASE = dict(
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
MEMBERS = [
    dict(max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    dict(max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
    dict(max_depth=7, subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, min_child_weight=3),
    dict(max_depth=6, subsample=0.9, colsample_bytree=0.6, reg_lambda=1.0),
    dict(max_depth=6, subsample=0.7, colsample_bytree=0.9, reg_lambda=0.5),
    dict(max_depth=8, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, min_child_weight=5,
         learning_rate=0.03),
    dict(max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0),
    dict(max_depth=6, subsample=1.0, colsample_bytree=0.5, reg_lambda=2.0, learning_rate=0.04),
    dict(max_depth=7, subsample=0.6, colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=2,
         learning_rate=0.04),
    dict(max_depth=6, subsample=0.85, colsample_bytree=0.75, reg_lambda=1.5, max_cat_to_onehot=1),
]


def _fit_member(params: dict, seed: int) -> xgb.XGBClassifier:
    p = dict(BASE, random_state=seed, **{k: v for k, v in params.items() if k != "seed"})
    X_tr, X_va, y_tr, y_va = train_test_split(X_all, y_all, test_size=0.15, random_state=seed, stratify=y_all)
    probe = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, **p)
    probe.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    n_rounds = int(probe.best_iteration) + 1
    m = xgb.XGBClassifier(n_estimators=int(n_rounds * 1.15) + 1, **p)
    m.fit(X_all, y_all)
    return m


X_all, y_all = prepare(train), to_y(train)
models = []
t0 = time.time()
for i, mp in enumerate(MEMBERS):
    models.append(_fit_member(mp, seed=SEED + i))
print(f"Training {len(models)} members: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
