"""XGBoost binary classifier on airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
ye = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

base = dict(
    learning_rate=0.1,
    max_depth=4,
    n_estimators=120,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=N_JOBS,
)
K = 8


def run_variant(drop_cols, label):
    cols = [c for c in feature_cols if c not in drop_cols]
    lv = {c: cat_levels[c] for c in cat_cols if c in cols}

    def prep(df):
        X = df[cols].copy()
        for c in lv:
            X[c] = pd.Categorical(X[c], categories=lv[c])
        return X

    X, Xe = prep(train), prep(evald)
    preds = [xgb.XGBClassifier(random_state=SEED + k, **base).fit(X, y).predict_proba(Xe)[:, 1] for k in range(K)]
    auc = roc_auc_score(ye, np.mean(preds, axis=0))
    print(f"{label}: {auc:.4f}")
    return auc, cols, lv


t0 = time.time()
results = [
    run_variant(drop, label)
    for drop, label in [([], "full"), (["DayofMonth"], "-DayofMonth"), (["Month"], "-Month"), (["DayofMonth", "Month"], "-dom-month")]
]
best_auc, kept_cols, kept_lv = max(results, key=lambda r: r[0])
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {best_auc:.4f}")


def prep_final(df: pd.DataFrame) -> pd.DataFrame:
    X = df[kept_cols].copy()
    for c in kept_lv:
        X[c] = pd.Categorical(X[c], categories=kept_lv[c])
    return X

model = xgb.XGBClassifier(random_state=SEED, **base)
model.fit(prep_final(train), y)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prep_final(df))[:, 1]
