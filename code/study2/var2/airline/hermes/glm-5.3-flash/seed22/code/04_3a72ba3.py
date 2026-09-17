"""XGBoost binary classifier: predict dep_delayed_15min (Y/N) on airline data.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Time-based validation: `python train.py -CV` (or CV_MODE=1) prints internal CV in addition to eval.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

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
    n_estimators=120,
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model = xgb.XGBClassifier(**PARAMS)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")

# --- development diagnostics (extra output; contract only requires the Eval AUC line) ---
if os.environ.get("CV_MODE") == "1":
    X = prepare(train)
    y = to_y(train)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    oof = np.zeros(len(y))
    for a, b in skf.split(X, y):
        m = xgb.XGBClassifier(**PARAMS)
        m.fit(X.iloc[a], y[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    print(f"Train-CV AUC: {roc_auc_score(y, oof):.4f}")
    ev = prepare(evald)
    print(f"Eval  AUC: {roc_auc_score(to_y(evald), model.predict_proba(ev)[:, 1]):.4f}")
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    print("Feature importance:", {feature_cols[i]: round(float(imp[i]), 3) for i in order})
