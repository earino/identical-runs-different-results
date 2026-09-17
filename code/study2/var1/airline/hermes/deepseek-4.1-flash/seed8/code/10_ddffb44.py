"""Sweep 10: wider depth x lr grids for the diversity ensemble."""
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

gridB = [dict(P, max_depth=dp, learning_rate=lr) for dp in (1, 2, 3, 4, 5, 6, 7, 8) for lr in (0.03, 0.06, 0.12)]
gridC = [dict(P, max_depth=dp, learning_rate=lr, n_estimators=n)
         for dp in (2, 3, 4, 5, 6, 7, 8) for lr, n in ((0.02, 900), (0.05, 400), (0.1, 250), (0.2, 150))]
recipes = {"gridB": gridB, "gridC": gridC}

results, raw = [], {}
t0 = time.time()
for name, params_list in recipes.items():
    preds = []
    for p in params_list:
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr, ytr)
        preds.append(m.predict_proba(Xev)[:, 1])
    raw[name] = np.array(preds)
    auc = roc_auc_score(yev, raw[name].mean(axis=0))
    results.append((auc, name))
    print(f"  AUC {auc:.4f}  {name:8s} n={len(params_list)}  [{time.time() - t0:.0f}s]", flush=True)

combo = np.vstack([raw["gridB"], raw["gridC"]])
print(f"  B+C combined({len(combo)}) AUC {roc_auc_score(yev, combo.mean(axis=0)):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")

results.sort(key=lambda r: -r[0])
best = results[0][1]
models = []
for p in recipes[best]:
    m = xgb.XGBClassifier(**p)
    m.fit(Xtr, ytr)
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print("BEST:", results[0])
print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
