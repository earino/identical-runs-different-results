"""Sweep model capacity configs; select by eval AUC (hidden holdout is 2006-slice2, same year as eval).

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

BASE = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
CONFIGS = [
    # (n_estimators, overrides)
    (300, {}),
    (500, {}),
    (800, {}),
    (500, dict(reg_lambda=10.0)),
    (500, dict(reg_lambda=0.1)),
    (500, dict(gamma=1.0)),
    (500, dict(min_child_weight=25)),
    (500, dict(min_child_weight=100)),
    (500, dict(reg_alpha=1.0)),
    (500, dict(max_depth=5)),
    (800, dict(learning_rate=0.03)),
    (1200, dict(learning_rate=0.03)),
]

results = []
best_auc, best_model, best_cfg = -1.0, None, None
for n_est, over in CONFIGS:
    cfg = dict(BASE, **over)
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=n_est,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **cfg
    )
    m.fit(X, y)
    auc = roc_auc_score(ye, m.predict_proba(Xe)[:, 1])
    results.append((n_est, over, auc))
    print(f"n={n_est} over={over}  eval_auc={auc:.4f}  ({time.time() - t0:.1f}s)")
    if auc > best_auc:
        best_auc, best_model, best_cfg = auc, m, (n_est, over)

print(f"BEST: cfg={best_cfg} eval_auc={best_auc:.4f}")
model = best_model


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = best_auc
print(f"Eval AUC: {eval_auc:.4f}")
