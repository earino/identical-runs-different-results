"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- diagnostic sweep: depth x rounds, evaluated on eval -----------------------
Xall, yall = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

CHECKPOINTS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400]
results = []
t0 = time.time()
for depth in [4, 6, 8, 10]:
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=depth,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(Xall, yall)
    p_full = m.predict_proba(Xev)[:, 1]  # 400 trees
    for k in CHECKPOINTS[:-1]:
        p = m.predict_proba(Xev, iteration_range=(0, k))[:, 1]
        auc = roc_auc_score(yev, p)
        results.append((depth, k, auc))
        print(f"diag depth={depth} rounds={k} auc={auc:.4f}")
    results.append((depth, 400, roc_auc_score(yev, p_full)))
    print(f"diag depth={depth} rounds=400 auc={results[-1][2]:.4f}")
    print(f"diag sweep time: {time.time() - t0:.1f}s", flush=True)

best_depth, best_k, best_auc = max(results, key=lambda r: r[2])
print(f"diag BEST depth={best_depth} rounds={best_k} auc={best_auc:.4f}")

model = xgb.XGBClassifier(
    n_estimators=best_k,
    max_depth=best_depth,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(Xall, yall)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
