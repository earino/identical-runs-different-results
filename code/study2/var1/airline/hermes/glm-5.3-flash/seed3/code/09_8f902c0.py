"""Seed-bagged XGBoost ensemble at the best single config (300 x d4, lr .05, ss/cs .8).

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

# --- feature layout (fitted on train only) ------------------------------------
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


X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
ye = to_y(evald)

# ensemble members: same capacity, diverse seeds + sampling
MEMBERS = [
    dict(random_state=42, subsample=0.8, colsample_bytree=0.8),
    dict(random_state=43, subsample=0.85, colsample_bytree=0.75),
    dict(random_state=44, subsample=0.75, colsample_bytree=0.85),
    dict(random_state=45, subsample=0.8, colsample_bytree=0.7),
    dict(random_state=46, subsample=0.9, colsample_bytree=0.8),
    dict(random_state=47, subsample=0.7, colsample_bytree=0.8),
    dict(random_state=48, subsample=0.8, colsample_bytree=0.9),
    dict(random_state=49, subsample=0.85, colsample_bytree=0.8),
]

preds = []
_models = []
t0 = time.time()
for i, over in enumerate(MEMBERS):
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **over
    )
    m.fit(X, y)
    p = m.predict_proba(Xe)[:, 1]
    auc = roc_auc_score(ye, p)
    print(f"member {i} over={over} eval_auc={auc:.4f}")
    preds.append(p)
    _models.append(m)
print(f"ensemble fit ({time.time() - t0:.1f}s)")

avg = np.mean(preds, axis=0)
eval_auc = roc_auc_score(ye, avg)
print(f"Ensemble AUC ({len(MEMBERS)} members): {eval_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in _models], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
