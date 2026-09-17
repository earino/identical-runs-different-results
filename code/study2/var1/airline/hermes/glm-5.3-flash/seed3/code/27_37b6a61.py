"""Drop-one feature re-check at the final recipe (d5, bylevel .9, bin512, lr03, n400), 3 seeds.

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

BASE = dict(n_estimators=400, max_depth=5, learning_rate=0.03, subsample=0.95, colsample_bytree=0.95, colsample_bylevel=0.9, max_bin=512)
SEEDS = [42, 43, 44]

FULL = list(X_all.columns)
VARIANTS = [("full", FULL)] + [(f"drop_{c}", [x for x in FULL if x != c]) for c in FULL]

summary = {}
t0 = time.time()
for name, feats in VARIANTS:
    preds = []
    aucs = []
    models = []
    for sd in SEEDS:
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=sd, n_jobs=N_JOBS, **BASE)
        m.fit(X_all[feats], y)
        p = m.predict_proba(Xe_all[feats])[:, 1]
        preds.append(p)
        aucs.append(roc_auc_score(ye, p))
        models.append((m, feats))
    avg_p = np.mean(preds, axis=0)
    avg_auc = float(roc_auc_score(ye, avg_p))
    summary[name] = (avg_auc, models, feats)
    print(f"{name}: seed_aucs={[round(a, 4) for a in aucs]} avg_pred_auc={avg_auc:.4f}")
print(f"({time.time() - t0:.1f}s)")

best_name = max(summary, key=lambda k: summary[k][0])
best_auc, models, best_feats = summary[best_name]
print(f"BEST: {best_name} avg_pred_auc={best_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)[best_feats]
    return np.mean([m.predict_proba(Xp)[:, 1] for m, _ in models], axis=0)


print(f"Eval AUC: {best_auc:.4f}")
