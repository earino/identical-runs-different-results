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
    dep = df["DepTime"]
    hour = dep // 100
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = hour * 60 + dep % 100
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
yall, yev = to_y(train), to_y(evald)
Xall, Xev = prepare(train), prepare(evald)

results = []
t0 = time.time()
for depth, lr, rounds, mcw, l2 in [
    (3, 0.1, 200, 1, 1),      # exp-5 best
    (3, 0.1, 200, 5, 1),
    (3, 0.1, 200, 20, 1),
    (3, 0.1, 200, 50, 1),
    (3, 0.1, 200, 20, 10),
    (3, 0.1, 200, 1, 10),
    (3, 0.05, 400, 1, 1),
    (3, 0.05, 400, 20, 1),
    (3, 0.05, 400, 20, 10),
    (3, 0.02, 1000, 20, 1),
    (4, 0.05, 400, 20, 1),
    (4, 0.05, 400, 20, 10),
    (4, 0.1, 200, 20, 1),
    (2, 0.1, 400, 1, 1),
]:
    m = xgb.XGBClassifier(
        n_estimators=rounds,
        max_depth=depth,
        learning_rate=lr,
        min_child_weight=mcw,
        reg_lambda=l2,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(Xall, yall)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, m, dict(depth=depth, lr=lr, rounds=rounds, mcw=mcw, l2=l2)))
    print(f"diag d={depth} lr={lr} r={rounds} mcw={mcw} l2={l2} auc={auc:.4f} t={time.time() - t0:.1f}s", flush=True)

best_auc, model, best_cfg = max(results, key=lambda r: r[0])
print(f"diag BEST {best_cfg} auc={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
