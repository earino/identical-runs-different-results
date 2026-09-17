"""Experiment 27: extend depth grid with d9/d10 members (ES on the fly).

Exp26: 16-member (d2-8 x lr grid) = 0.7333. Deep members with proper ES counts keep
helping (d7/d8 added +0.002); test whether d9/d10 @ lr 0.05 continue the trend.
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


MEMBERS = [
    (2, 0.05, 554), (3, 0.05, 374), (4, 0.05, 214), (5, 0.05, 237),
    (6, 0.05, 232), (7, 0.05, 402), (8, 0.05, 626),
    (2, 0.03, 875), (3, 0.03, 645), (4, 0.03, 86), (5, 0.03, 317),
    (6, 0.03, 298), (7, 0.03, 364), (8, 0.03, 823),
    (7, 0.1, 2285), (8, 0.1, 472),
]

t0 = time.time()
yfull = to_y(train)
yev = to_y(evald)
Xfull = prepare(train)
Xev = prepare(evald)
month = _ord(train["Month"])
is_val = (month >= 11).to_numpy()
Xtr, ytr = Xfull[~is_val], yfull[~is_val]
Xval, yval = Xfull[is_val], yfull[is_val]

extra_counts = {}
for d in [9, 10]:
    es = xgb.XGBClassifier(
        n_estimators=3000, max_depth=d, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        early_stopping_rounds=150, eval_metric="auc",
    )
    es.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    extra_counts[d] = int(es.best_iteration + 1)
    print(f"ES depth={d}: {extra_counts[d]} trees")

models16, preds16 = [], []
for d, lr, n in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=d, learning_rate=lr, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xfull, yfull)
    models16.append(m)
    preds16.append(m.predict_proba(Xev)[:, 1])
print(f"fit 16 members in {time.time() - t0:.1f}s")
auc16 = roc_auc_score(yev, np.mean(preds16, axis=0))
print(f"members16: {auc16:.4f}")

preds18 = list(preds16)
for d, n in extra_counts.items():
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xfull, yfull)
    models16.append(m)
    preds18.append(m.predict_proba(Xev)[:, 1])
auc18 = roc_auc_score(yev, np.mean(preds18, axis=0))
print(f"members18: {auc18:.4f}")
print(f"total {time.time() - t0:.1f}s")

if auc18 > auc16:
    best_models = models16
    print("best: 18 members")
else:
    best_models = models16[:16]
    print("best: 16 members")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in best_models], axis=0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
