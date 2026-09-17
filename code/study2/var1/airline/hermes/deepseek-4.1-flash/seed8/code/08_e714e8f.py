"""Sweep 8: larger ensemble recipes + rank averaging."""
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

P = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
         n_estimators=300, learning_rate=0.05, min_child_weight=10, max_depth=4)
ALL = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
RED = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]


def make_prepare(cols):
    ccols = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
    lv = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ccols}
    o = train["Origin"].value_counts()
    d = train["Dest"].value_counts()
    cc = train["UniqueCarrier"].value_counts()

    def prepare(df):
        X = df[cols].copy()
        for c in ccols:
            X[c] = pd.Categorical(X[c], categories=lv[c])
        X["freq_origin"] = np.log1p(df["Origin"].map(o).fillna(0).to_numpy())
        X["freq_dest"] = np.log1p(df["Dest"].map(d).fillna(0).to_numpy())
        X["freq_carrier"] = np.log1p(df["UniqueCarrier"].map(cc).fillna(0).to_numpy())
        return X

    return prepare


PREP_RED = make_prepare(RED)
PREP_ALL = make_prepare(ALL)
Xtr, Xev = PREP_RED(train), PREP_RED(evald)

wide = [dict(P, max_depth=d, learning_rate=lr) for d in (3, 4, 5) for lr in (0.03, 0.05, 0.1)]
recipes = {
    "wide9": (wide, PREP_RED),
    "wide18": ([dict(P, max_depth=d, learning_rate=lr, min_child_weight=m)
                for d in (3, 4, 5) for lr in (0.03, 0.05, 0.1) for m in (5, 10)], PREP_RED),
    "deepmix": ([dict(P, max_depth=d, learning_rate=lr) for d in (2, 3, 4, 5, 6) for lr in (0.04, 0.08)], PREP_RED),
    "roundsmix": (wide + [dict(P, max_depth=4, learning_rate=0.05, n_estimators=n) for n in (150, 600)], PREP_RED),
    "featmix": (wide + [dict(P, max_depth=d, learning_rate=0.05) for d in (3, 4)]
                + [dict(P, max_depth=4, learning_rate=0.05)], PREP_ALL),
}

results = []
raw = {}
t0 = time.time()
for name, (params_list, prep) in recipes.items():
    Xt, Xe = prep(train), prep(evald)
    preds = []
    for p in params_list:
        m = xgb.XGBClassifier(**p)
        m.fit(Xt, ytr)
        preds.append(m.predict_proba(Xe)[:, 1])
    raw[name] = np.array(preds)
    auc = roc_auc_score(yev, raw[name].mean(axis=0))
    results.append((auc, name))
    print(f"  AUC {auc:.4f}  {name:10s} n={len(params_list)}  [{time.time() - t0:.0f}s]", flush=True)

# rank averaging on the best recipe
results.sort(key=lambda r: -r[0])
best = results[0][1]
R = np.array([pd.Series(p).rank().to_numpy() for p in raw[best]])
rank_auc = roc_auc_score(yev, R.mean(axis=0))
print(f"  rank-avg({best}) AUC {rank_auc:.4f}")
print("BEST:", results[0])
print(f"Training time: {time.time() - t0:.1f}s")

params_list, prep = recipes[best]
models = []
for p in params_list:
    m = xgb.XGBClassifier(**p)
    m.fit(prep(train), ytr)
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prep(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
