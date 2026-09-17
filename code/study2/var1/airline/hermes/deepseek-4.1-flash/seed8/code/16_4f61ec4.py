"""Sweep 13: fit depths 1-16 x lr {0.04,0.16}, then compare subsets (AUC vs runtime trade-off)."""
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


Xtr, Xev = prepare(train), prepare(evald)
GRID = [(d, lr) for d in range(1, 17) for lr in (0.04, 0.16)]

t0 = time.time()
models, preds = [], []
for d, lr in GRID:
    m = xgb.XGBClassifier(max_depth=d, learning_rate=lr, n_estimators=300, min_child_weight=10,
                          tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
    m.fit(Xtr, ytr)
    models.append(m)
    preds.append(m.predict_proba(Xev)[:, 1])
P = np.array(preds)
print(f"fit+predict done [{time.time() - t0:.0f}s]", flush=True)

idx = {p: i for i, p in enumerate(GRID)}
SUBSETS = {
    "full d1-16 x2": list(range(32)),
    "d1-16 lr.16": [idx[(d, 0.16)] for d in range(1, 17)],
    "d1-16 lr.04": [idx[(d, 0.04)] for d in range(1, 17)],
    "d9-16 x2": [idx[(d, lr)] for d in range(9, 17) for lr in (0.04, 0.16)],
    "d13-16 x2": [idx[(d, lr)] for d in range(13, 17) for lr in (0.04, 0.16)],
    "d1-12 x2": [idx[(d, lr)] for d in range(1, 13) for lr in (0.04, 0.16)],
    "d5-16 x2": [idx[(d, lr)] for d in range(5, 17) for lr in (0.04, 0.16)],
}

best = (0.0, None)
for name, sel in SUBSETS.items():
    auc = roc_auc_score(yev, P[sel].mean(axis=0))
    print(f"  AUC {auc:.4f}  {name:16s} n={len(sel)}", flush=True)
    if auc > best[0]:
        best = (auc, name)
z = np.clip(P, 1e-6, 1 - 1e-6)
print(f"  AUC {roc_auc_score(yev, np.log(z / (1 - z)).mean(axis=0)):.4f}  full logit-avg", flush=True)
print(f"Training time: {time.time() - t0:.1f}s")
print(f"SELECTED {best[1]} ({best[0]:.4f})")

sel = SUBSETS[best[1]]
sel_models = [models[i] for i in sel]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in sel_models], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
