"""Experiment 4: early stopping on a temporally separated split (2005 months 11-12 as val).

Motivation: exp2/3 showed random-split val AUC 0.7595 while 2006 eval dropped to 0.6949
(baseline 30 trees: 0.7141) -> overfitting to 2005-specific patterns. A temporal val split
should pick a tree count that generalizes across the year gap.
Features: baseline-style native categoricals for all 6 cat cols + dep_min + dayofyear.
"""
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)


def _ord(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_missing"] = dt.isna().astype(float)
    dtf = dt.fillna(0)
    h = (dtf // 100).astype(float) % 24
    X["dep_min"] = h * 60 + dtf % 100
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
    month = _ord(df["Month"]).fillna(1).clip(1, 12).astype(int)
    X["dayofyear"] = CUM_DAYS[month - 1] + _ord(df["DayofMonth"]).fillna(15)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    max_depth=6, learning_rate=0.1, tree_method="hist", enable_categorical=True,
    random_state=SEED, n_jobs=N_JOBS,
)

t0 = time.time()
month = _ord(train["Month"])
is_val = (month >= 11).to_numpy()
Xtr, ytr = prepare(train[~is_val]), to_y(train[~is_val])
Xval, yval = prepare(train[is_val]), to_y(train[is_val])
print(f"temporal split: train={(~is_val).sum()} val={is_val.sum()}")

es_model = xgb.XGBClassifier(
    n_estimators=3000, early_stopping_rounds=100, eval_metric="auc", **PARAMS
)
es_model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
best_n = int(es_model.best_iteration + 1)
print(f"ES picked {best_n} trees, val AUC {es_model.best_score:.4f} ({time.time() - t0:.1f}s)")

# retrain on ALL of 2005 with the chosen count (no ES)
model = xgb.XGBClassifier(n_estimators=best_n, **PARAMS)
model.fit(prepare(train), to_y(train))
print(f"retrained on full train in {time.time() - t0:.1f}s total")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
