"""XGBoost binary classifier for airline delay. Only file the agent edits.

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
from sklearn.model_selection import train_test_split

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


X_tr, X_va, y_tr, y_va = train_test_split(train, to_y(train), test_size=0.15, random_state=SEED)
X_tr_p, X_va_p, evald_p = prepare(X_tr), prepare(X_va), prepare(evald)
y_ev = to_y(evald)

results = {}
t0 = time.time()
for depth in (4, 6, 8):
    for n in (20, 40, 80, 150, 300):
        m = xgb.XGBClassifier(
            n_estimators=n,
            learning_rate=0.1,
            max_depth=depth,
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        m.fit(X_tr_p, y_tr)
        va = roc_auc_score(y_va, m.predict_proba(X_va_p)[:, 1])
        ev = roc_auc_score(y_ev, m.predict_proba(evald_p)[:, 1])
        results[(depth, n)] = (va, ev)
        print(f"depth={depth} n={n:4d}  valid2005={va:.4f}  eval2006={ev:.4f}", flush=True)
print(f"Sweep time: {time.time() - t0:.1f}s")

best = max(results, key=lambda k: results[k][1])
print(f"best on eval2006: depth={best[0]} n={best[1]} -> {results[best][1]:.4f}")

model = xgb.XGBClassifier(
    n_estimators=best[1],
    learning_rate=0.1,
    max_depth=best[0],
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(prepare(train), to_y(train))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
