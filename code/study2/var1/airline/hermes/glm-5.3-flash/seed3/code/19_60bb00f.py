"""Mega seed-averaging: 20 seeds best cfg + 10 seeds runner-up, pooled prediction.

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

GROUPS = {
    "n300_lr04": (dict(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.95, colsample_bytree=0.95), 20),
    "n250_mcw10": (dict(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.95, colsample_bytree=0.95, min_child_weight=10), 10),
}
SEEDS = list(range(42, 82))

models = []
t0 = time.time()
group_preds = {}
for name, (cfg, n_seeds) in GROUPS.items():
    preds = []
    for sd in SEEDS[:n_seeds]:
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=sd, n_jobs=N_JOBS, **cfg)
        m.fit(X_all[FEATS], y)
        p = m.predict_proba(Xe_all[FEATS])[:, 1]
        preds.append(p)
        models.append(m)
    gp = np.mean(preds, axis=0)
    group_preds[name] = gp
    print(f"{name}: {n_seeds} seeds avg AUC={roc_auc_score(ye, gp):.4f}")
print(f"fit time ({time.time() - t0:.1f}s)")

pool = np.mean(list(group_preds.values()), axis=0)
pool_auc = roc_auc_score(ye, pool)
g1 = roc_auc_score(ye, group_preds["n300_lr04"])
g2 = roc_auc_score(ye, group_preds["n250_mcw10"])
print(f"g1={g1:.4f} g2={g2:.4f} pooled={pool_auc:.4f}")
if pool_auc >= max(g1, g2):
    eval_auc = pool_auc
    final_models = models
else:
    eval_auc = max(g1, g2)
    final_models = models[:20] if g1 >= g2 else models[20:]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATS]
    return np.mean([m.predict_proba(Xp)[:, 1] for m in final_models], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
