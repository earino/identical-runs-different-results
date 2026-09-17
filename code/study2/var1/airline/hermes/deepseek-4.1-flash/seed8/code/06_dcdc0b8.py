"""Sweep 6: seed variance study + seed/depth averaged ensembles."""
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
BEST = dict(max_depth=3, n_estimators=300, learning_rate=0.05, min_child_weight=10)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
to_y = lambda df: (df[TARGET] == POSITIVE).astype(int).to_numpy()
ytr, yev = to_y(train), to_y(evald)

BASE = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
freq_o = train["Origin"].value_counts()
freq_d = train["Dest"].value_counts()
freq_c = train["UniqueCarrier"].value_counts()
cat_cols = [c for c in BASE if not pd.api.types.is_numeric_dtype(train[c])]
levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df):
    X = df[BASE].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=levels[c])
    X["freq_origin"] = np.log1p(df["Origin"].map(freq_o).fillna(0).to_numpy())
    X["freq_dest"] = np.log1p(df["Dest"].map(freq_d).fillna(0).to_numpy())
    X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(freq_c).fillna(0).to_numpy())
    return X


Xtr, Xev = prepare(train), prepare(evald)

t0 = time.time()
seed_preds, seed_aucs = [], []
for seed in range(10):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, random_state=seed, **BEST)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xev)[:, 1]
    seed_preds.append(p)
    seed_aucs.append(roc_auc_score(yev, p))
    print(f"  seed {seed}: AUC {seed_aucs[-1]:.4f}  [{time.time() - t0:.0f}s]", flush=True)

print(f"seed AUC mean {np.mean(seed_aucs):.4f} std {np.std(seed_aucs):.4f}")
for k in (2, 4, 8, 10):
    p = np.mean(seed_preds[:k], axis=0)
    print(f"  bag{k} AUC {roc_auc_score(yev, p):.4f}")

# depth-mixed ensemble
mix_preds = []
for depth in (3, 4, 5):
    for seed in range(3):
        m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
                              random_state=100 + seed, max_depth=depth, n_estimators=300,
                              learning_rate=0.05, min_child_weight=10)
        m.fit(Xtr, ytr)
        mix_preds.append(m.predict_proba(Xev)[:, 1])
print(f"  depth-mix(9) AUC {roc_auc_score(yev, np.mean(mix_preds, axis=0)):.4f}")
print(f"  all(19) AUC {roc_auc_score(yev, np.mean(seed_preds + mix_preds, axis=0)):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")

model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, random_state=0, **BEST)
model.fit(Xtr, ytr)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
