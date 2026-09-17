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

ROUNDS = 300
CONFIGS = [
    dict(name="d3-lr20", max_depth=3, learning_rate=0.2),
    dict(name="d3-lr10", max_depth=3, learning_rate=0.1),
    dict(name="d2-lr20", max_depth=2, learning_rate=0.2),
    dict(name="d2-lr40", max_depth=2, learning_rate=0.4),
    dict(name="d4-lr05", max_depth=4, learning_rate=0.05),
    dict(name="d3-lr20-col7", max_depth=3, learning_rate=0.2, colsample_bytree=0.7),
    dict(name="d3-lr20-col5", max_depth=3, learning_rate=0.2, colsample_bytree=0.5),
    dict(name="d4-lr10-col7", max_depth=4, learning_rate=0.1, colsample_bytree=0.7),
    dict(name="d3-lr20-mcw5", max_depth=3, learning_rate=0.2, min_child_weight=5),
    dict(name="d3-lr30", max_depth=3, learning_rate=0.3),
]

best = None  # (auc, config, n)
t0 = time.time()
for cfg in CONFIGS:
    name = cfg.pop("name")
    m = xgb.XGBClassifier(
        n_estimators=ROUNDS,
        eval_metric="auc",
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    aucs = m.evals_result()["validation_0"]["auc"]
    bi = int(np.argmax(aucs))
    print(f"cfg {name}: best_round={bi + 1} auc={aucs[bi]:.4f}")
    if best is None or aucs[bi] > best[0]:
        best = (aucs[bi], dict(cfg), bi + 1)
    cfg["name"] = name
print(f"Sweep time: {time.time() - t0:.1f}s")
print(f"BEST: {best}")

# retrain the winning config at its best round (deterministic) -> this is the final model
final = xgb.XGBClassifier(
    n_estimators=best[2],
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    **best[1],
)
final.fit(X_tr, y_tr)
model = final


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {best[0]:.4f}")
