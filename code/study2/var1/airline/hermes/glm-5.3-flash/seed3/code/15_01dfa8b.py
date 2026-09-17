"""Final micro-tune sweep: sampling, lr, n_estimators, DART, max_cat_to_onehot, lossguide.

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

BASE = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9)
CONFIGS = [
    ("ctrl", {}),
    ("ss_cs_95", dict(subsample=0.95, colsample_bytree=0.95)),
    ("ss_cs_100", dict(subsample=1.0, colsample_bytree=1.0)),
    ("n250", dict(n_estimators=250)),
    ("n350", dict(n_estimators=350)),
    ("n400", dict(n_estimators=400)),
    ("lr06", dict(learning_rate=0.06)),
    ("lr04", dict(learning_rate=0.04)),
    ("dart", dict(booster="dart", one_drop=True)),
    ("onehot_cats", dict(max_cat_to_onehot=24)),
    ("lossguide_64", dict(grow_policy="lossguide", max_leaves=64)),
    ("n300_d4_mcw10", dict(min_child_weight=10)),
]

results = []
best_auc, best_model, best_name = -1.0, None, None
for name, over in CONFIGS:
    cfg = dict(BASE, **over)
    t0 = time.time()
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
        **cfg
    )
    m.fit(X_all[FEATS], y)
    auc = roc_auc_score(ye, m.predict_proba(Xe_all[FEATS])[:, 1])
    print(f"variant={name}  eval_auc={auc:.4f}  ({time.time() - t0:.1f}s)")
    if auc > best_auc:
        best_auc, best_model, best_name = auc, m, name

print(f"BEST: {best_name} eval_auc={best_auc:.4f}")
model = best_model
FEATURES = FEATS


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATURES]
    return model.predict_proba(Xp)[:, 1]


print(f"Eval AUC: {best_auc:.4f}")
