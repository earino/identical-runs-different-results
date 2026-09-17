"""Sweep experiment 3: feature ablations under the best config (depth3/300/lr.05/mcw10)."""
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
BEST = dict(max_depth=3, n_estimators=300, learning_rate=0.05, min_child_weight=10)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
ALL = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]

FEATURE_SETS = {
    "all": ALL,
    "-Origin": [c for c in ALL if c != "Origin"],
    "-Dest": [c for c in ALL if c != "Dest"],
    "-UniqueCarrier": [c for c in ALL if c != "UniqueCarrier"],
    "-Distance": [c for c in ALL if c != "Distance"],
    "-calendar": [c for c in ALL if c not in ("Month", "DayofMonth", "DayOfWeek")],
    "-Origin-Dest": [c for c in ALL if c not in ("Origin", "Dest")],
    "DepTime+Distance": ["DepTime", "Distance"],
    "DepTime+Distance+Carrier+DOW": ["DepTime", "Distance", "UniqueCarrier", "DayOfWeek"],
}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)


def make_prepare(cols):
    cat_cols = [c for c in cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
    levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

    def prepare(df):
        X = df[cols].copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=levels[c])
        return X

    return prepare


results = []
t0 = time.time()
for name, cols in FEATURE_SETS.items():
    prep = make_prepare(cols)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **BEST)
    m.fit(prep(train), ytr)
    auc = roc_auc_score(yev, m.predict_proba(prep(evald))[:, 1])
    results.append((auc, name))
    print(f"  AUC {auc:.4f}  {name:32s} [{time.time() - t0:.0f}s]", flush=True)

results.sort(key=lambda r: -r[0])
print("BEST:", results[0])
print(f"Training time: {time.time() - t0:.1f}s")

best_name = results[0][1]
prep = make_prepare(FEATURE_SETS[best_name])
model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **BEST)
model.fit(prep(train), ytr)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prep(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
