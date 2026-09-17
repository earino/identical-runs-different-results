"""Sweep 7: diversity ensembles (depth x subsample x colsample x seed)."""
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

BASE_P = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS,
              n_estimators=300, learning_rate=0.05, min_child_weight=10)


def recipes():
    r = {}
    r["depthmix"] = [dict(BASE_P, max_depth=d) for d in (2, 3, 4, 5)]
    r["depthmix_sub"] = [dict(BASE_P, max_depth=d, subsample=0.8, colsample_bytree=0.8, random_state=s)
                         for d in (3, 4, 5) for s in (0, 1)]
    r["depthmix_sub_all"] = [dict(BASE_P, max_depth=d, subsample=0.75, colsample_bytree=0.75, random_state=s)
                             for d in (2, 3, 4, 5) for s in (0, 1, 2)]
    r["depthmix_seedsub"] = [dict(BASE_P, max_depth=d, subsample=0.7, colsample_bytree=0.7, random_state=s)
                             for d in (3, 4, 5, 6) for s in (0, 1, 2)]
    r["depthmix_wide"] = [dict(BASE_P, max_depth=d, learning_rate=lr)
                          for d in (3, 4, 5) for lr in (0.03, 0.05, 0.1)]
    return r


results = []
t0 = time.time()
for name, params_list in recipes().items():
    preds = []
    for p in params_list:
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr, ytr)
        preds.append(m.predict_proba(Xev)[:, 1])
    auc = roc_auc_score(yev, np.mean(preds, axis=0))
    results.append((auc, name, params_list))
    print(f"  AUC {auc:.4f}  {name:22s} n={len(params_list)}  [{time.time() - t0:.0f}s]", flush=True)

results.sort(key=lambda r: -r[0])
print("BEST:", results[0][0], results[0][1])
print(f"Training time: {time.time() - t0:.1f}s")

best_params = results[0][2]
models = []
for p in best_params:
    m = xgb.XGBClassifier(**p)
    m.fit(Xtr, ytr)
    models.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
