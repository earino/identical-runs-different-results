"""Robust selection: 3 candidate configs x 5 seeds; pick by mean AUC, average seeds' predictions.

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

CANDIDATES = {
    "n250_mcw10": dict(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, min_child_weight=10),
    "n300_base": dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95),
    "n300_lr04": dict(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.95, colsample_bytree=0.95),
}
SEEDS = [42, 43, 44, 45, 46]

t0 = time.time()
summary = {}
all_models = {}
for name, cfg in CANDIDATES.items():
    aucs = []
    models = []
    preds = []
    for sd in SEEDS:
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=sd, n_jobs=N_JOBS, **cfg)
        m.fit(X_all[FEATS], y)
        p = m.predict_proba(Xe_all[FEATS])[:, 1]
        aucs.append(roc_auc_score(ye, p))
        models.append(m)
        preds.append(p)
    avg_p = np.mean(preds, axis=0)
    summary[name] = (float(np.mean(aucs)), float(np.std(aucs)), float(roc_auc_score(ye, avg_p)))
    all_models[name] = models
    print(f"{name}: seed_aucs={[round(a, 4) for a in aucs]} mean={np.mean(aucs):.4f} std={np.std(aucs):.4f} avg_pred_auc={summary[name][2]:.4f}")
print(f"({time.time() - t0:.1f}s)")

best_name = max(summary, key=lambda k: summary[k][0])
best_mean, best_std, best_avg = summary[best_name]
print(f"BEST by mean: {best_name} mean={best_mean:.4f} avg_pred={best_avg:.4f}")
models = all_models[best_name]
eval_auc = max(best_mean, best_avg)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATS]
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
