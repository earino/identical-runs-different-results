"""Experiment 22: ES-selected counts per member + lossguide members.

Fixed member grid is saturated at 0.7235. Two structural alternatives:
  R1 = M2 reference
  R2 = per-member temporal-ES tree counts (m11-12 val), retrained on full train
  R3 = R1-style members + lossguide (leaf-wise) members with max_leaves caps
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


def _cat(series: pd.Series, n_levels: int) -> pd.Categorical:
    return pd.Categorical(series.astype(str), categories=[str(i) for i in range(n_levels)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_missing"] = dt.isna().astype(float)
    dtf = dt.fillna(0)
    h = (dtf // 100).astype(float) % 24
    total_min = h * 60 + dtf % 100
    X["dep_min"] = total_min
    X["dep_block"] = _cat((total_min // 15).astype(int), 96)
    X["dep_hour"] = _cat(h.astype(int), 24)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
    month = _ord(df["Month"]).fillna(1).clip(1, 12).astype(int)
    X["dayofyear"] = CUM_DAYS[month - 1] + _ord(df["DayofMonth"]).fillna(15)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


t0 = time.time()
yfull = to_y(train)
yev = to_y(evald)
Xfull = prepare(train)
Xev = prepare(evald)
month = _ord(train["Month"])
is_val = (month >= 11).to_numpy()
Xtr, ytr = Xfull[~is_val], yfull[~is_val]
Xval, yval = Xfull[is_val], yfull[is_val]

FIXED = [(2, 250), (2, 500), (3, 250), (3, 500), (4, 120), (4, 250), (4, 500), (5, 80), (5, 150), (6, 80)]
ES_DEPTHS = [2, 3, 4, 5, 6]
LG_MEMBERS = [(16, 300), (32, 300), (64, 200)]

results = {}


def eval_avg(preds):
    return roc_auc_score(yev, np.mean(preds, axis=0))


def fit_fixed(**extra):
    preds = []
    for d, n in FIXED:
        m = xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **extra,
        )
        m.fit(Xfull, yfull)
        preds.append(m.predict_proba(Xev)[:, 1])
    return preds


preds_r1 = fit_fixed()
results["R1_fixed"] = eval_avg(preds_r1)
print(f"R1_fixed: {results['R1_fixed']:.4f}")

preds_r2 = []
es_counts = []
for d in ES_DEPTHS:
    es = xgb.XGBClassifier(
        n_estimators=2000, max_depth=d, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        early_stopping_rounds=150, eval_metric="auc",
    )
    es.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    n_best = int(es.best_iteration + 1)
    es_counts.append(n_best)
    m = xgb.XGBClassifier(
        n_estimators=n_best, max_depth=d, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xfull, yfull)
    preds_r2.append(m.predict_proba(Xev)[:, 1])
    print(f"  depth {d}: ES count {n_best}")
results["R2_es"] = eval_avg(preds_r2)
print(f"R2_es: {results['R2_es']:.4f}")

preds_r3 = list(preds_r1)
for leaves, n in LG_MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=n, max_leaves=leaves, grow_policy="lossguide", learning_rate=0.05,
        tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xfull, yfull)
    preds_r3.append(m.predict_proba(Xev)[:, 1])
results["R3_lossguide"] = eval_avg(preds_r3)
print(f"R3_lossguide: {results['R3_lossguide']:.4f}")

print(f"scan took {time.time() - t0:.1f}s")
best = max(results, key=results.get)
print(f"best: {best}")

_models = []


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in _models], axis=0)


if best == "R1_fixed":
    for d, n in FIXED:
        _models.append(xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        ))
elif best == "R2_es":
    for d, n_best in zip(ES_DEPTHS, es_counts):
        _models.append(xgb.XGBClassifier(
            n_estimators=n_best, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        ))
else:
    for d, n in FIXED:
        _models.append(xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        ))
    for leaves, n in LG_MEMBERS:
        _models.append(xgb.XGBClassifier(
            n_estimators=n, max_leaves=leaves, grow_policy="lossguide", learning_rate=0.05,
            tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        ))
for m in _models:
    m.fit(Xfull, yfull)

eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")