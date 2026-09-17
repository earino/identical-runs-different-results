"""Sweep 12: fit depths 1-12 x 4 lrs (48 models), then compare sub-ensembles / averaging rules."""
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
to_y = lambda df: (df[TARGET] == POSITIVE).astype(int).to_numpy()
ytr, yev = to_y(train), to_y(evald)

FEATURES = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
cat_cols = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
freq_tables = {c: train[c].value_counts() for c in ("Origin", "Dest", "UniqueCarrier")}


def prepare(df):
    X = df[FEATURES].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    X["freq_origin"] = np.log1p(df["Origin"].map(freq_tables["Origin"]).fillna(0).to_numpy())
    X["freq_dest"] = np.log1p(df["Dest"].map(freq_tables["Dest"]).fillna(0).to_numpy())
    X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(freq_tables["UniqueCarrier"]).fillna(0).to_numpy())
    return X


BASE = dict(n_estimators=300, min_child_weight=10, tree_method="hist",
            enable_categorical=True, n_jobs=N_JOBS)
LRS = (0.02, 0.04, 0.08, 0.16)
GRID = [(d, lr) for d in range(1, 13) for lr in LRS]

Xtr, Xev = prepare(train), prepare(evald)
t0 = time.time()
params_list, models, preds = [], [], []
for d, lr in GRID:
    m = xgb.XGBClassifier(max_depth=d, learning_rate=lr, **BASE)
    m.fit(Xtr, ytr)
    params_list.append((d, lr))
    models.append(m)
    preds.append(m.predict_proba(Xev)[:, 1])
    if len(models) % 12 == 0:
        print(f"  fitted {len(models)} models [{time.time() - t0:.0f}s]", flush=True)
P = np.array(preds)
print(f"fit done [{time.time() - t0:.0f}s]", flush=True)


def auc_of(idx, logit=False, weights=None):
    sub = P[idx]
    w = np.ones(len(idx)) if weights is None else weights
    if logit:
        z = np.clip(sub, 1e-6, 1 - 1e-6)
        z = np.log(z / (1 - z))
        s = np.average(z, axis=0, weights=w)
        return roc_auc_score(yev, s)
    return roc_auc_score(yev, np.average(sub, axis=0, weights=w))


idx = {p: i for i, p in enumerate(params_list)}
sel_gridX = [idx[(d, lr)] for d in range(1, 11) for lr in LRS]
sel_all = list(range(len(params_list)))
sel_low = [idx[(d, lr)] for d in range(1, 13) for lr in (0.02, 0.08)]
sel_high = [idx[(d, lr)] for d in range(1, 13) for lr in (0.04, 0.16)]
sel_deep = [idx[(d, lr)] for d in range(5, 13) for lr in LRS]

for name, sel in [("d1-10x4 (ref)", sel_gridX), ("d1-12x4", sel_all), ("d1-12 lr.02/.08", sel_low),
                  ("d1-12 lr.04/.16", sel_high), ("d5-12x4", sel_deep)]:
    print(f"  AUC {auc_of(sel):.4f}  {name}  n={len(sel)}", flush=True)
print(f"  AUC {auc_of(sel_all, logit=True):.4f}  d1-12x4 logit-avg  n={len(sel_all)}", flush=True)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
