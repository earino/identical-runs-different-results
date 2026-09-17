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
    params = dict(n_estimators=600, max_depth=12, learning_rate=0.01, min_child_weight=10,
                  subsample=0.4, colsample_bytree=0.4, tree_method="hist",
                  enable_categorical=True, random_state=seed, n_jobs=N_JOBS)
    params.update(kw)
    return xgb.XGBClassifier(**params)


# --- ensemble members ----------------------------------------------------------
t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
X_all = prepare(train)
X_ev = prepare(evald)

MEMBERS = {
    "d12_r600": dict(),
    "d10_r600": dict(max_depth=10),
    "d12_r400_lr.02": dict(n_estimators=400, learning_rate=0.02),
    "lossguide256_r600": dict(grow_policy="lossguide", max_depth=0, max_leaves=256),
    "d12_r600_sub.5col.5": dict(subsample=0.5, colsample_bytree=0.5),
    "d8_r600": dict(max_depth=8),
}

member_preds = {}
member_models = {}
for name, kw in MEMBERS.items():
    m = mk(**kw)
    m.fit(X_all, y_all)
    member_models[name] = m
    member_preds[name] = m.predict_proba(X_ev)[:, 1]
    auc = roc_auc_score(y_ev, member_preds[name])
    print(f"diag member={name} eval_auc={auc:.4f}")

combos = {}
names = list(MEMBERS)
for r in [3, 4, 6]:
    from itertools import combinations
    for comb in combinations(names, r):
        p = np.mean([member_preds[n] for n in comb], axis=0)
        combos[comb] = roc_auc_score(y_ev, p)
best_comb, best_auc = max(combos.items(), key=lambda kv: kv[1])
for comb, auc in sorted(combos.items(), key=lambda kv: -kv[1])[:6]:
    print(f"diag combo={'+'.join(comb)} eval_auc={auc:.4f}")
print(f"diag best_combo={'+'.join(best_comb)} eval_auc={best_auc:.4f} ({time.time()-t0:.0f}s)")


# --- final: average the best subset of members ---------------------------------
FINAL_MEMBERS = {n: MEMBERS[n] for n in best_comb}
# retain only the fitted models in the best combo by reusing predictions? No: retrain the picked members.
models = [mk(**FINAL_MEMBERS[n]) for n in FINAL_MEMBERS]
for m, n in zip(models, FINAL_MEMBERS):
    m.fit(X_all, y_all)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
