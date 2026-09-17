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
    "d5_bl90_n400": dict(n_estimators=400, max_depth=5, colsample_bylevel=0.9),
    "d5_ctrl_n400": dict(n_estimators=400, max_depth=5),
    "d6_bl90_n500": dict(n_estimators=500, max_depth=6, colsample_bylevel=0.9),
    "d4_bl90_n500": dict(n_estimators=500, max_depth=4, colsample_bylevel=0.9),
}
SEEDS = list(range(42, 50))

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
full = np.mean([group_preds[n] for n in names], axis=0)
auc_full = roc_auc_score(ye, full)
print(f"full ensemble (4 cfg x 8 seeds): {auc_full:.4f}")
for drop in names:
    rest = [group_preds[n] for n in names if n != drop]
    print(f"  without {drop}: {roc_auc_score(ye, np.mean(rest, axis=0)):.4f}")

eval_auc = auc_full
final_names = names
best_single = max(names, key=lambda n: roc_auc_score(ye, group_preds[n]))
auc_best_single = float(roc_auc_score(ye, group_preds[best_single]))
if auc_best_single > auc_full:
    eval_auc = auc_best_single
    final_names = [best_single]
    print(f"single group wins: {best_single} {auc_best_single:.4f}")

final_models = [m for n in final_names for m in group_models[n]]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[FEATS]
    return np.mean([m.predict_proba(Xp)[:, 1] for m in final_models], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
