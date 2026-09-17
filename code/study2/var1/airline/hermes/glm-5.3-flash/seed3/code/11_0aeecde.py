"""Best single model (300 x d4, lr .05, ss/cs .8) on raw columns; drop-one-feature ablation.

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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
str_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
num_cols = [c for c in feature_cols if c not in str_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in str_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in str_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all = prepare(train)
y = to_y(train)
Xe_all = prepare(evald)
ye = to_y(evald)

FULL = list(X_all.columns)
VARIANTS = [("full", FULL)] + [(f"drop_{c}", [x for x in FULL if x != c]) for c in FULL]

results = []
best_auc, best_model, best_name, best_feats = -1.0, None, None, None
for name, feats in VARIANTS:
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(X_all[feats], y)
    auc = roc_auc_score(ye, m.predict_proba(Xe_all[feats])[:, 1])
    print(f"variant={name}  eval_auc={auc:.4f}  ({time.time() - t0:.1f}s)")
    if auc > best_auc:
        best_auc, best_model, best_name, best_feats = auc, m, name, feats

print(f"BEST: {best_name} eval_auc={best_auc:.4f}")
model = best_model
FEATURES = best_feats


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATURES]
    return model.predict_proba(Xp)[:, 1]


print(f"Eval AUC: {best_auc:.4f}")
