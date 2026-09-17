"""Sweep 15: single-model probes of encoding / regularisation variants (fast)."""
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


def prepare(df, freqs=True):
    X = df[BASE].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    if freqs:
        X["freq_origin"] = np.log1p(df["Origin"].map(freq_tables["Origin"]).fillna(0).to_numpy())
        X["freq_dest"] = np.log1p(df["Dest"].map(freq_tables["Dest"]).fillna(0).to_numpy())
        X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(freq_tables["UniqueCarrier"]).fillna(0).to_numpy())
    return X


Xtr, Xev = prepare(train), prepare(evald)
Xtr_nf, Xev_nf = prepare(train, freqs=False), prepare(evald, freqs=False)
COM = dict(max_depth=12, learning_rate=0.16, n_estimators=300, min_child_weight=10,
           tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)

CASES = [
    ("ref d12", Xtr, Xev, {}),
    ("onehot cats", Xtr, Xev, dict(max_cat_to_onehot=1000)),
    ("no freq feats", Xtr_nf, Xev_nf, {}),
    ("600 rounds", Xtr, Xev, dict(n_estimators=600)),
    ("mcw1", Xtr, Xev, dict(min_child_weight=1)),
    ("depth20", Xtr, Xev, dict(max_depth=20)),
    ("gamma5", Xtr, Xev, dict(gamma=5.0)),
    ("l2=20,l1=2", Xtr, Xev, dict(reg_lambda=20.0, reg_alpha=2.0)),
    ("colsample0.8", Xtr, Xev, dict(colsample_bytree=0.8)),
    ("interaction_constraints", Xtr, Xev, dict(interaction_constraints=[[0, 1, 2, 3, 4, 5, 6, 7]])),
]

t0 = time.time()
for name, Xt, Xe, over in CASES:
    m = xgb.XGBClassifier(**dict(COM, **over))
    m.fit(Xt, ytr)
    print(f"  AUC {roc_auc_score(yev, m.predict_proba(Xe)[:, 1]):.4f}  {name:24s} {over}  [{time.time() - t0:.0f}s]", flush=True)
print(f"Training time: {time.time() - t0:.1f}s")

m = xgb.XGBClassifier(**COM)
m.fit(Xtr, ytr)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
