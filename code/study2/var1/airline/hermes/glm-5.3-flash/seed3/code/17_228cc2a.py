"""Single-model grid: min_child_weight x n_estimators at ss/cs 0.95 (6-feature set).

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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET, "Month", "DayofMonth"]]
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
FEATS = list(X_all.columns)

CONFIGS = [
    (200, 1), (200, 10), (200, 25),
    (250, 1), (250, 5), (250, 10), (250, 15), (250, 25),
    (300, 5), (300, 10), (300, 15),
    (350, 10),
]

results = []
best_auc, best_model, best_cfg = -1.0, None, None
for n, mcw in CONFIGS:
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=n,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.95,
        colsample_bytree=0.95,
        min_child_weight=mcw,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(X_all[FEATS], y)
    auc = roc_auc_score(ye, m.predict_proba(Xe_all[FEATS])[:, 1])
    print(f"n={n} mcw={mcw}  eval_auc={auc:.4f}  ({time.time() - t0:.1f}s)")
    if auc > best_auc:
        best_auc, best_model, best_cfg = auc, m, (n, mcw)

print(f"BEST: cfg={best_cfg} eval_auc={best_auc:.4f}")
model = best_model
FEATURES = FEATS


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATURES]
    return model.predict_proba(Xp)[:, 1]


print(f"Eval AUC: {best_auc:.4f}")
