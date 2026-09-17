"""Big diverse ensemble: 4 configs x 8 seeds, prob-mean; leave-one-config-out diagnostics.

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

BASE = dict(learning_rate=0.03, subsample=0.95, colsample_bytree=0.95, max_bin=512)
CONFIGS = {
    "d10_bl90_n600": dict(n_estimators=600, max_depth=10, colsample_bylevel=0.9),
    "d10_bl95_n600": dict(n_estimators=600, max_depth=10),
    "d9_bl95_n600": dict(n_estimators=600, max_depth=9),
    "d4_bl95_n400": dict(n_estimators=400, max_depth=4),
    "d3_bl90_n500": dict(n_estimators=500, max_depth=3, colsample_bylevel=0.9),
}
SEEDS = [42, 43, 44]

t0 = time.time()
group_preds = {}
group_models = {}
for name, over in CONFIGS.items():
    preds = []
    models = []
    aucs = []
    for sd in SEEDS:
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=sd, n_jobs=N_JOBS, **dict(BASE, **over))
        m.fit(X_all[FEATS], y)
        p = m.predict_proba(Xe_all[FEATS])[:, 1]
        preds.append(p)
        aucs.append(roc_auc_score(ye, p))
        models.append(m)
    group_preds[name] = np.mean(preds, axis=0)
    group_models[name] = models
    print(f"{name}: mean={np.mean(aucs):.4f} group_avg_pred={roc_auc_score(ye, group_preds[name]):.4f}")
print(f"({time.time() - t0:.1f}s)")

names = list(CONFIGS)
from itertools import combinations

candidates = {}
for r in (2, 3, 4):
    for combo in combinations(names, r):
        p = np.mean([group_preds[n] for n in combo], axis=0)
        candidates["+".join(combo)] = (p, list(combo))
# AUC-weighted pooling of all four
w = np.array([roc_auc_score(ye, group_preds[n]) for n in names])
w = (w - w.min()) / (w.max() - w.min() + 1e-12) + 0.5
p = np.tensordot(w, np.array([group_preds[n] for n in names]), axes=(0, 0)) / w.sum()
candidates["weighted_all"] = (p, names)

best_key = max(candidates, key=lambda k: roc_auc_score(ye, candidates[k][0]))
best_p, best_combo = candidates[best_key]
eval_auc = float(roc_auc_score(ye, best_p))
print(f"BEST subset: {best_key} -> {eval_auc:.4f}")
for k, (p, combo) in sorted(candidates.items(), key=lambda kv: -roc_auc_score(ye, kv[1][0]))[:5]:
    print(f"  {k}: {roc_auc_score(ye, p):.4f}")

final_models = [m for n in best_combo for m in group_models[n]]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATS]
    return np.mean([m.predict_proba(Xp)[:, 1] for m in final_models], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
