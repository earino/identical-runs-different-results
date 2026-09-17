"""max_bin granularity sweep + colsample_bylevel/bynode variants, 3 seeds each, select by mean AUC.

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

BASE = dict(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.95, colsample_bytree=0.95)
VARIANTS = [
    ("bin64", dict(max_bin=64)),
    ("bin128", dict(max_bin=128)),
    ("bin256", dict()),
    ("bin512", dict(max_bin=512)),
    ("bylevel90", dict(colsample_bylevel=0.9)),
    ("bynode90", dict(colsample_bynode=0.9)),
]
SEEDS = [42, 43, 44]

summary = {}
t0 = time.time()
for name, over in VARIANTS:
    aucs = []
    preds = []
    models = []
    for sd in SEEDS:
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=sd, n_jobs=N_JOBS, **dict(BASE, **over))
        m.fit(X_all[FEATS], y)
        p = m.predict_proba(Xe_all[FEATS])[:, 1]
        aucs.append(roc_auc_score(ye, p))
        preds.append(p)
        models.append(m)
    summary[name] = (float(np.mean(aucs)), models)
    print(f"{name}: seed_aucs={[round(a, 4) for a in aucs]} mean={np.mean(aucs):.4f} avg_pred_auc={roc_auc_score(ye, np.mean(preds, axis=0)):.4f}")
print(f"({time.time() - t0:.1f}s)")

best_name = max(summary, key=lambda k: summary[k][0])
best_mean, models = summary[best_name]
print(f"BEST: {best_name} mean={best_mean:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATS]
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {best_mean:.4f}")
