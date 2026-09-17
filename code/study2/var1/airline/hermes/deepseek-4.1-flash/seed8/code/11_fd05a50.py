"""Sweep 11: very wide depth x lr grids, plus subsample/seed diversity on top."""
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

RED = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
P = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
         n_estimators=300, learning_rate=0.05, min_child_weight=10, max_depth=4)

ccols = [c for c in RED if not pd.api.types.is_numeric_dtype(train[c])]
lv = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ccols}
o = train["Origin"].value_counts()
d = train["Dest"].value_counts()
cc = train["UniqueCarrier"].value_counts()


def prepare(df):
    X = df[RED].copy()
    for c in ccols:
        X[c] = pd.Categorical(X[c], categories=lv[c])
    X["freq_origin"] = np.log1p(df["Origin"].map(o).fillna(0).to_numpy())
    X["freq_dest"] = np.log1p(df["Dest"].map(d).fillna(0).to_numpy())
    X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(cc).fillna(0).to_numpy())
    return X


Xtr, Xev = prepare(train), prepare(evald)

gridX = [dict(P, max_depth=dp, learning_rate=lr) for dp in range(1, 11) for lr in (0.02, 0.04, 0.08, 0.16)]
extra = [dict(P, max_depth=dp, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=s)
         for dp in (2, 3, 4, 5, 6, 7, 9) for s in (0, 1)]

t0 = time.time()
preds = []
for p in gridX:
    m = xgb.XGBClassifier(**p)
    m.fit(Xtr, ytr)
    preds.append(m.predict_proba(Xev)[:, 1])
preds = np.array(preds)
print(f"  gridX({len(gridX)}) AUC {roc_auc_score(yev, preds.mean(axis=0)):.4f}  [{time.time() - t0:.0f}s]", flush=True)

extra_preds = []
for p in extra:
    m = xgb.XGBClassifier(**p)
    m.fit(Xtr, ytr)
    extra_preds.append(m.predict_proba(Xev)[:, 1])
extra_preds = np.array(extra_preds)
print(f"  extra({len(extra)}) AUC {roc_auc_score(yev, extra_preds.mean(axis=0)):.4f}  [{time.time() - t0:.0f}s]", flush=True)
print(f"  gridX+extra({len(gridX) + len(extra)}) AUC {roc_auc_score(yev, np.vstack([preds, extra_preds]).mean(axis=0)):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")

models = []
for p in gridX:
    m = xgb.XGBClassifier(**p)
    m.fit(Xtr, ytr)
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
