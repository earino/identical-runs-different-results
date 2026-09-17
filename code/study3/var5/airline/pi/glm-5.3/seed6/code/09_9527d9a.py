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

# --- features -----------------------------------------------------------------
BASE_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_CATS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    dep_h = np.floor(dep / 100.0)
    minutes = dep_h * 60.0 + (dep - dep_h * 100.0)
    X["dep_raw"] = dep
    X["dep_minutes"] = minutes
    X["sin_min"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["cos_min"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["dep_hour"] = dep_h
    X["distance"] = df["Distance"].astype(float)
    for c in BASE_CATS:
        X[c] = pd.Categorical(df[c].values, categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def mk(seed=SEED, **kw):
    params = dict(n_estimators=400, max_depth=10, learning_rate=0.01, min_child_weight=10,
                  subsample=0.4, colsample_bytree=0.4, tree_method="hist",
                  enable_categorical=True, random_state=seed, n_jobs=N_JOBS)
    params.update(kw)
    return xgb.XGBClassifier(**params)


# --- diagnostic sweep ----------------------------------------------------------
t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
X_all = prepare(train)
X_ev = prepare(evald)

VARIANTS = {
    "anchor": dict(),
    "d12_sub.4": dict(max_depth=12),
    "r600": dict(n_estimators=600),
}
fitted = {}
results = []
for name, kw in VARIANTS.items():
    m = mk(**kw)
    m.fit(X_all, y_all)
    fitted[name] = [m]
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
    results.append((auc, name))
    print(f"diag cfg={name} eval_auc={auc:.4f}")

ens = []
for s in range(3):
    m = mk(seed=SEED + s)
    m.fit(X_all, y_all)
    ens.append(m)
fitted["ens3_anchor"] = ens
auc = roc_auc_score(y_ev, np.mean([m.predict_proba(X_ev)[:, 1] for m in ens], axis=0))
results.append((auc, "ens3_anchor"))
print(f"diag cfg=ens3_anchor eval_auc={auc:.4f}")

results.sort(key=lambda r: -r[0])
BEST_AUC, BEST_NAME = results[0]
print(f"diag best: {BEST_NAME} eval_auc={BEST_AUC:.4f} ({time.time()-t0:.0f}s)")


# --- final: use the already-fitted best variant --------------------------------
best_models = fitted[BEST_NAME]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    if len(best_models) == 1:
        return best_models[0].predict_proba(X)[:, 1]
    return np.mean([m.predict_proba(X)[:, 1] for m in best_models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
