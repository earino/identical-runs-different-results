"""Sweep experiment: evaluate several XGBoost configs on eval.csv in one run (baseline features)."""
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

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

CONFIGS = []
for depth in (4, 6, 8):
    for n in (30, 60, 150):
        CONFIGS.append(dict(max_depth=depth, n_estimators=n, learning_rate=0.1))
for depth in (3, 4, 6):
    CONFIGS.append(dict(max_depth=depth, n_estimators=300, learning_rate=0.05, min_child_weight=10))
for mcw in (1, 20):
    CONFIGS.append(dict(max_depth=6, n_estimators=30, learning_rate=0.1, min_child_weight=mcw, subsample=0.7, colsample_bytree=0.7))

results = []
t0 = time.time()
for cfg in CONFIGS:
    params = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    params.update(cfg)
    m = xgb.XGBClassifier(**params)
    m.fit(Xtr, ytr)
    auc = roc_auc_score(yev, m.predict_proba(Xev)[:, 1])
    results.append((auc, cfg))
    print(f"  AUC {auc:.4f}  {cfg}   [{time.time() - t0:.0f}s]", flush=True)

results.sort(key=lambda r: -r[0])
print("BEST:", results[0])
print(f"Training time: {time.time() - t0:.1f}s")

best_cfg = results[0][1]
model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **best_cfg)
model.fit(Xtr, ytr)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
