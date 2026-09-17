"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
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


# --- model --------------------------------------------------------------------
def make_model(depth: int, n: int, lr: float, seed: int, sub: float = 1.0, col: float = 1.0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n,
        max_depth=depth,
        learning_rate=lr,
        subsample=sub,
        colsample_bytree=col,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )

# --- Caruana greedy ensemble selection on an INTERNAL validation split (no eval leakage) ---
from sklearn.model_selection import train_test_split

X_all = prepare(train)
y_all = to_y(train)
X_in, X_val, y_in, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)

CANDIDATES = [
    (4, 60, 0.10, 42, 1.0, 1.0),
    (6, 30, 0.10, 7, 1.0, 1.0),
    (4, 120, 0.05, 123, 0.8, 0.8),
    (5, 45, 0.10, 2024, 0.9, 0.9),
    (3, 100, 0.10, 5, 1.0, 1.0),
    (4, 80, 0.08, 11, 0.7, 0.7),
    (6, 40, 0.07, 13, 0.8, 0.9),
    (5, 60, 0.06, 17, 1.0, 0.8),
    (4, 60, 0.10, 555, 0.85, 0.85),
    (5, 50, 0.10, 556, 0.9, 0.9),
    (6, 35, 0.10, 557, 0.9, 0.9),
    (4, 100, 0.06, 558, 1.0, 0.9),
    (3, 120, 0.12, 559, 0.9, 1.0),
    (7, 25, 0.10, 560, 0.9, 0.9),
    (5, 70, 0.08, 561, 0.8, 0.8),
    (4, 60, 0.12, 562, 1.0, 1.0),
]

t0 = time.time()
val_preds, cand_val_auc = [], []
for cfg in CANDIDATES:
    m = make_model(*cfg)
    m.fit(X_in, y_in)
    p = m.predict_proba(X_val)[:, 1]
    val_preds.append(p)
    cand_val_auc.append(roc_auc_score(y_val, p))

N_ROUNDS = 40
chosen = [int(np.argmax(cand_val_auc))]  # start from the best single candidate
run_sum = val_preds[chosen[0]].copy()
for _ in range(N_ROUNDS - 1):
    best_auc, best_j = -1.0, -1
    for j in range(len(CANDIDATES)):
        auc = roc_auc_score(y_val, (run_sum + val_preds[j]) / (len(chosen) + 1))
        if auc > best_auc:
            best_auc, best_j = auc, j
    chosen.append(best_j)
    run_sum = run_sum + val_preds[best_j]
sel_auc = roc_auc_score(y_val, run_sum / len(chosen))
print(f"Cand AUCs: {[round(a, 4) for a in cand_val_auc]}")
print(f"Selected {len(chosen)} members (with repeats), internal-val AUC {sel_auc:.4f}")

# retrain the selected configs on the FULL training set + add a fixed curated ensemble for blending
models = []
from collections import Counter

for idx, cnt in Counter(chosen).items():
    cfg = CANDIDATES[idx]
    for _ in range(cnt):
        models.append(make_model(*cfg))
FIXED8 = [
    (4, 60, 0.10, 42, 1.0, 1.0),
    (6, 30, 0.10, 7, 1.0, 1.0),
    (4, 120, 0.05, 123, 0.8, 0.8),
    (5, 45, 0.10, 2024, 0.9, 0.9),
    (3, 100, 0.10, 5, 1.0, 1.0),
    (4, 80, 0.08, 11, 0.7, 0.7),
    (6, 40, 0.07, 13, 0.8, 0.9),
    (5, 60, 0.06, 17, 1.0, 0.8),
]
models += [make_model(*c) for c in FIXED8]
X_tr, y_tr = X_all, y_all
t0 = time.time()
for m in models:
    m.fit(X_tr, y_tr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
