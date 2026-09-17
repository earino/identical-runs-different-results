"""Sweep 16: ensemble arms over regularization settings (min_child_weight / L2) at depths 1-16."""
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
DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16]
COM = dict(learning_rate=0.16, n_estimators=300, tree_method="hist",
           enable_categorical=True, n_jobs=N_JOBS)


def ensemble(over, depths=DEPTHS):
    preds = []
    for d in depths:
        m = xgb.XGBClassifier(max_depth=d, **dict(COM, **over))
        m.fit(Xtr, ytr)
        preds.append(m.predict_proba(Xev)[:, 1])
    return np.array(preds)


ARMS = {
    "mcw10 (ref)": dict(min_child_weight=10),
    "mcw1": dict(min_child_weight=1),
    "l2=20,l1=2": dict(min_child_weight=10, reg_lambda=20.0, reg_alpha=2.0),
    "mcw1+l2=20,l1=2": dict(min_child_weight=1, reg_lambda=20.0, reg_alpha=2.0),
}

t0 = time.time()
res = {}
for name, over in ARMS.items():
    P = ensemble(over)
    res[name] = (roc_auc_score(yev, P.mean(axis=0)), P)
    print(f"  AUC {res[name][0]:.4f}  {name:16s} [{time.time() - t0:.0f}s]", flush=True)

mix = np.vstack([res["mcw10 (ref)"][1], res["mcw1"][1], res["l2=20,l1=2"][1], res["mcw1+l2=20,l1=2"][1]])
print(f"  AUC {roc_auc_score(yev, mix.mean(axis=0)):.4f}  mixed-4-arms n={len(mix)}")
print(f"Training time: {time.time() - t0:.1f}s")

win = max(res, key=lambda k: res[k][0])
print("WINNER", win)
models = []
for d in DEPTHS:
    m = xgb.XGBClassifier(max_depth=d, **dict(COM, **ARMS[win]))
    m.fit(Xtr, ytr)
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
