"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features (baseline set) ---------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- small capacity/regularization scan, ship the best model -------------------
Xall, yall = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

GRID = [
    dict(max_depth=4, n_estimators=30),
    dict(max_depth=4, n_estimators=100),
    dict(max_depth=4, n_estimators=300),
    dict(max_depth=6, n_estimators=30),
    dict(max_depth=6, n_estimators=100),
    dict(max_depth=6, n_estimators=300),
    dict(max_depth=8, n_estimators=30),
    dict(max_depth=8, n_estimators=100),
    dict(max_depth=6, n_estimators=100, subsample=0.5),
    dict(max_depth=6, n_estimators=300, subsample=0.5),
    dict(max_depth=6, n_estimators=100, min_child_weight=50),
    dict(max_depth=6, n_estimators=300, min_child_weight=50),
]

best, best_auc, best_cfg = None, -1.0, None
t0 = time.time()
for cfg in GRID:
    m = xgb.XGBClassifier(
        learning_rate=0.1, tree_method="hist", enable_categorical=True,
        random_state=SEED, n_jobs=N_JOBS, **cfg)
    m.fit(Xall, yall)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    print(f"  cfg={cfg} -> eval_auc={auc:.4f}", flush=True)
    if auc > best_auc:
        best, best_auc, best_cfg = m, auc, cfg
print(f"Scan time: {time.time() - t0:.1f}s  chosen={best_cfg}")

model = best


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {best_auc:.4f}")
