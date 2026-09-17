"""Sweep 5: remaining variants on the reduced feature set (no route categorical)."""
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
to_y = lambda df: (df[TARGET] == POSITIVE).astype(int).to_numpy()
ytr, yev = to_y(train), to_y(evald)

BASE = ["DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
freq_o = train["Origin"].value_counts()
freq_d = train["Dest"].value_counts()
freq_c = train["UniqueCarrier"].value_counts()


def make_prepare(cols, extra=None):
    extra = extra or {}
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(train[c])]
    levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

    def prepare(df):
        X = df[cols].copy()
        for c in cat_cols:
            X[c] = pd.Categorical(X[c], categories=levels[c])
        for name, fn in extra.items():
            X[name] = fn(df)
        return X

    return prepare


FREQOD = {
    "freq_origin": lambda df: np.log1p(df["Origin"].map(freq_o).fillna(0).to_numpy()),
    "freq_dest": lambda df: np.log1p(df["Dest"].map(freq_d).fillna(0).to_numpy()),
}
FREQALL = dict(FREQOD, freq_carrier=lambda df: np.log1p(df["UniqueCarrier"].map(freq_c).fillna(0).to_numpy()))

CASES = [
    ("base", BASE, {}, {}),
    ("base+freqOD", BASE, FREQOD, {}),
    ("base+freqOD+carrier", BASE, FREQALL, {}),
    ("base+freqOD, maxbin64", BASE, FREQOD, dict(max_bin=64)),
    ("base+freqOD, mcw30", BASE, FREQOD, dict(min_child_weight=30)),
    ("base+freqOD, gamma1", BASE, FREQOD, dict(gamma=1.0)),
    ("base+freqOD, sub0.7", BASE, FREQOD, dict(subsample=0.7, colsample_bytree=0.7)),
    ("base+freqOD, noDist", [c for c in BASE if c != "Distance"], FREQOD, {}),
    ("base+freqOD, 600t", BASE, FREQOD, dict(n_estimators=600)),
    ("base+freqOD, depth4", BASE, FREQOD, dict(max_depth=4, n_estimators=600)),
]

results = []
t0 = time.time()
for name, cols, extra, over in CASES:
    prep = make_prepare(cols, extra)
    params = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    params.update(BEST)
    params.update(over)
    m = xgb.XGBClassifier(**params)
    m.fit(prep(train), ytr)
    auc = roc_auc_score(yev, m.predict_proba(prep(evald))[:, 1])
    results.append((auc, name, cols, extra, over))
    print(f"  AUC {auc:.4f}  {name:24s} {over}  [{time.time() - t0:.0f}s]", flush=True)

results.sort(key=lambda r: -r[0])
print("BEST:", results[0][0], results[0][1], results[0][4])
print(f"Training time: {time.time() - t0:.1f}s")

_, best_name, best_cols, best_extra, best_over = results[0]
prep = make_prepare(best_cols, best_extra)
params = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **BEST)
params.update(best_over)
model = xgb.XGBClassifier(**params)
model.fit(prep(train), ytr)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prep(df))[:, 1]


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
