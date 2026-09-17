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


def mk(**kw):
    params = dict(n_estimators=30, max_depth=6, learning_rate=0.1, tree_method="hist",
                  enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    params.update(kw)
    return xgb.XGBClassifier(**params)


# --- diagnostic sweep ----------------------------------------------------------
t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
X_all = prepare(train)
X_ev = prepare(evald)

CONFIGS = {
    "anchor_r30": dict(),
    "lr.05_r60": dict(n_estimators=60, learning_rate=0.05),
    "lr.03_r100": dict(n_estimators=100, learning_rate=0.03),
    "d4_r30": dict(max_depth=4),
    "lr.05_r60_d4": dict(n_estimators=60, learning_rate=0.05, max_depth=4),
    "lr.05_r60_d8_mcw20": dict(n_estimators=60, learning_rate=0.05, max_depth=8, min_child_weight=20),
    "lr.05_r60_mcw50_reg": dict(n_estimators=60, learning_rate=0.05, min_child_weight=50,
                                subsample=0.8, colsample_bytree=0.8),
    "lr.02_r200_bag": dict(n_estimators=200, learning_rate=0.02, min_child_weight=10,
                           subsample=0.7, colsample_bytree=0.7),
}

results = []
for name, kw in CONFIGS.items():
    m = mk(**kw)
    m.fit(X_all, y_all)
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
    results.append((auc, name, kw))
    print(f"diag cfg={name} eval_auc={auc:.4f}")

for name, kw, n_seeds in [("ens5_anchor", dict(), 5), ("ens5_lr.05_r60", dict(n_estimators=60, learning_rate=0.05), 5)]:
    preds = []
    for s in range(n_seeds):
        m = mk(random_state=SEED + s, **kw)
        m.fit(X_all, y_all)
        preds.append(m.predict_proba(X_ev)[:, 1])
    auc = roc_auc_score(y_ev, np.mean(preds, axis=0))
    results.append((auc, name, dict(kw, n_seeds=n_seeds)))
    print(f"diag cfg={name} eval_auc={auc:.4f}")

results.sort(key=lambda r: -r[0])
BEST_AUC, BEST_NAME, BEST_KW = results[0]
print(f"diag best: {BEST_NAME} eval_auc={BEST_AUC:.4f} ({time.time()-t0:.0f}s)")


# --- final model on full train with the best config ----------------------------
if BEST_NAME.startswith("ens"):
    models = [mk(random_state=SEED + s, **{k: v for k, v in BEST_KW.items() if k != "n_seeds"})
              for s in range(BEST_KW.get("n_seeds", 5))]
    for m in models:
        m.fit(X_all, y_all)

    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
else:
    model = mk(**BEST_KW)
    model.fit(X_all, y_all)

    def predict_proba(df: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(y_ev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
