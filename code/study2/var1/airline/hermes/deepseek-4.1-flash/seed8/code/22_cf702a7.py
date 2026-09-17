"""Sweep 18: cheaper mcw0 configurations (fewer rounds) + sub-ensemble choice for runtime safety."""
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

BASE = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
cat_cols = [c for c in BASE if not pd.api.types.is_numeric_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
freq_tables = {c: train[c].value_counts() for c in ("Origin", "Dest", "UniqueCarrier")}


def prepare(df):
    X = df[BASE].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    X["freq_origin"] = np.log1p(df["Origin"].map(freq_tables["Origin"]).fillna(0).to_numpy())
    X["freq_dest"] = np.log1p(df["Dest"].map(freq_tables["Dest"]).fillna(0).to_numpy())
    X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(freq_tables["UniqueCarrier"]).fillna(0).to_numpy())
    return X


Xtr, Xev = prepare(train), prepare(evald)
COM = dict(learning_rate=0.16, n_estimators=200, min_child_weight=0,
           tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
DEPTHS = list(range(1, 17))

t0 = time.time()
models, preds = [], []
for d in DEPTHS:
    m = xgb.XGBClassifier(max_depth=d, **COM)
    m.fit(Xtr, ytr)
    models.append(m)
    preds.append(m.predict_proba(Xev)[:, 1])
P = np.array(preds)
print(f"fit ({time.time() - t0:.0f}s)", flush=True)

SUBSETS = {
    "d1-16 n200": list(range(16)),
    "d1-12 n200": list(range(12)),
    "d1-14 n200": list(range(14)),
    "d3-16 n200": list(range(2, 16)),
}
best = (0.0, None)
for name, sel in SUBSETS.items():
    auc = roc_auc_score(yev, P[sel].mean(axis=0))
    print(f"  AUC {auc:.4f}  {name} n={len(sel)}", flush=True)
    if auc > best[0]:
        best = (auc, name)
print(f"Training time: {time.time() - t0:.1f}s  SELECTED {best[1]}")

sel = SUBSETS[best[1]]
sel_models = [models[i] for i in sel]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in sel_models], axis=0)


print(f"Eval AUC: {best[0]:.4f}")
