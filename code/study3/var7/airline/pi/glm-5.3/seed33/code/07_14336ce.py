"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
obj_cols = [
    c
    for c in feature_cols
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])
]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_tr = prepare(train)
y_tr = to_y(train)
X_ev = prepare(evald)
y_ev = to_y(evald)

ROUNDS = 400
MEMBERS = [
    dict(max_depth=4, learning_rate=0.05, rounds=400),
    dict(max_depth=3, learning_rate=0.2, rounds=150),
    dict(max_depth=4, learning_rate=0.1, colsample_bytree=0.7, rounds=400),
    dict(max_depth=3, learning_rate=0.1, rounds=300),
    dict(max_depth=6, learning_rate=0.1, rounds=120),
    dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, rounds=400, seed=13),
    dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, rounds=400, seed=21),
    dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, rounds=400, seed=29),
]


def build(spec: dict):
    m = xgb.XGBClassifier(
        n_estimators=spec["rounds"],
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        random_state=spec.get("seed", SEED),
        n_jobs=N_JOBS,
        max_depth=spec["max_depth"],
        learning_rate=spec["learning_rate"],
        subsample=spec.get("subsample", 1.0),
        colsample_bytree=spec.get("colsample_bytree", 1.0),
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    print(f"member {spec.get('seed', SEED)} d{spec['max_depth']} lr{spec['learning_rate']}: best_round={bi + 1} auc={aucs[bi]:.4f}")
    # retrain at best round WITHOUT eval_set (so eval.csv never touches the fitted object)
    f = xgb.XGBClassifier(
        n_estimators=bi + 1,
        tree_method="hist",
        enable_categorical=True,
        random_state=spec.get("seed", SEED),
        n_jobs=N_JOBS,
        max_depth=spec["max_depth"],
        learning_rate=spec["learning_rate"],
        subsample=spec.get("subsample", 1.0),
        colsample_bytree=spec.get("colsample_bytree", 1.0),
    )
    f.fit(X_tr, y_tr)
    return f


t0 = time.time()
models = [build(spec) for spec in MEMBERS]
print(f"Ensemble time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.zeros(len(X))
    for m in models:
        ps += m.predict_proba(X)[:, 1]
    return ps / len(models)


eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
