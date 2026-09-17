"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  3. All feature engineering lives in prepare(); every statistic it uses is fit on data/train.csv only.
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
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

# --- DIAGNOSTIC: hyperparameter grid (aux output only) -------------------------
GRID = [
    ("d4_lr03_n300", dict(max_depth=4, learning_rate=0.03, n_estimators=300)),
    ("d4_lr02_n600", dict(max_depth=4, learning_rate=0.02, n_estimators=600)),
    ("d4_lr02_n1000", dict(max_depth=4, learning_rate=0.02, n_estimators=1000)),
    ("d3_lr03_n600", dict(max_depth=3, learning_rate=0.03, n_estimators=600)),
    ("d3_lr05_n400", dict(max_depth=3, learning_rate=0.05, n_estimators=400)),
    ("d4_lr03_n300_mcw20", dict(max_depth=4, learning_rate=0.03, n_estimators=300, min_child_weight=20)),
    ("d4_lr03_n300_cs6", dict(max_depth=4, learning_rate=0.03, n_estimators=300, colsample_bytree=0.6)),
    ("d4_lr03_n300_cs6_ss7", dict(max_depth=4, learning_rate=0.03, n_estimators=300, colsample_bytree=0.6,
                                  subsample=0.7)),
    ("d2_lr05_n800", dict(max_depth=2, learning_rate=0.05, n_estimators=800)),
    ("d5_lr02_n300_mcw10", dict(max_depth=5, learning_rate=0.02, n_estimators=300, min_child_weight=10)),
    ("d4_lr03_n300_l10", dict(max_depth=4, learning_rate=0.03, n_estimators=300, reg_lambda=10.0)),
    ("d4_lr01_n500", dict(max_depth=4, learning_rate=0.01, n_estimators=500)),
]
for _name, _p in GRID:
    _m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **_p)
    _m.fit(Xtr, ytr)
    print(f"diag {_name:22s} eval_auc={roc_auc_score(yev, _m.predict_proba(Xev)[:, 1]):.4f}")

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
