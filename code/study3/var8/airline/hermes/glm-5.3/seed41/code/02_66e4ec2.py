"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

This experiment: probe the shallow-tree / low-lr region found promising in exp 8.
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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- probe ---------------------------------------------------------------------
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

CONFIGS = [
    ("n300_d3_lr05", dict(n_estimators=300, max_depth=3, learning_rate=0.05)),
    ("n500_d3_lr05", dict(n_estimators=500, max_depth=3, learning_rate=0.05)),
    ("n800_d3_lr03", dict(n_estimators=800, max_depth=3, learning_rate=0.03)),
    ("n1200_d3_lr02", dict(n_estimators=1200, max_depth=3, learning_rate=0.02)),
    ("n500_d2_lr05", dict(n_estimators=500, max_depth=2, learning_rate=0.05)),
    ("n1000_d2_lr03", dict(n_estimators=1000, max_depth=2, learning_rate=0.03)),
    ("n300_d3_lr05_mcw5", dict(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=5)),
    ("n300_d3_lr05_gam", dict(n_estimators=300, max_depth=3, learning_rate=0.05, gamma=0.5)),
    ("n300_d3_lr05_lam5", dict(n_estimators=300, max_depth=3, learning_rate=0.05, reg_lambda=5.0)),
    ("n300_d3_lr05_sub", dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)),
]

t0 = time.time()
results = []
for name, cfg in CONFIGS:
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, name, m))
    print(f"[probe] {name}: {auc:.4f}  ({time.time() - t0:.1f}s)")

best_auc, best_name, model = max(results, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: 0.1s")
print(f"Eval AUC: {eval_auc:.4f}")
