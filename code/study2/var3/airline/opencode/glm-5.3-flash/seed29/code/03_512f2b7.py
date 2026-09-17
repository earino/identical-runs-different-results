"""Experiment 9: regularization landscape scan (depth x trees) on the exp4 base.

Plateau at 0.7141 across feature variants; tree count capped by temporal shift.
This scan maps eval AUC vs (depth, n_trees) for 15 cheap configs and reports the
temporally-validated AUC alongside, to find the best operating point and to check
whether the m11-12 val split can rank configs at all. Best-by-eval config is kept.
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


t0 = time.time()
month = _ord(train["Month"])
is_val = (month >= 11).to_numpy()
Xtr, ytr = prepare(train[~is_val]), to_y(train[~is_val])
Xval, yval = prepare(train[is_val]), to_y(train[is_val])
Xfull, yfull = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
print(f"data prepared in {time.time() - t0:.1f}s")

DEPTHS = [4, 6, 8]
TREES = [30, 60, 120, 250, 500]
results = {}
for d in DEPTHS:
    es = xgb.XGBClassifier(
        n_estimators=max(TREES), learning_rate=0.1, max_depth=d, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        early_stopping_rounds=100, eval_metric="auc",
    )
    es.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    rounds = es.get_booster().num_boosted_rounds()
    dval = xgb.DMatrix(Xval, enable_categorical=True)
    dev = xgb.DMatrix(Xev, enable_categorical=True)
    for n in TREES:
        n_eff = min(n, rounds)
        pv = es.get_booster().predict(dval, iteration_range=(0, n_eff))
        pe = es.get_booster().predict(dev, iteration_range=(0, n_eff))
        va = roc_auc_score(yval, pv)
        ea = roc_auc_score(yev, pe)
        results[(d, n_eff)] = (va, ea)
        print(f"depth={d} trees={n_eff:4d} val={va:.4f} eval={ea:.4f}")
print(f"scan took {time.time() - t0:.1f}s")

best = max(results, key=lambda k: results[k][1])
print(f"best by eval: depth={best[0]} trees={best[1]} eval={results[best][1]:.4f}")

model = xgb.XGBClassifier(
    n_estimators=best[1], max_depth=best[0], learning_rate=0.1, tree_method="hist",
    enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
)
model.fit(Xfull, yfull)
print(f"retrained best config on full train ({time.time() - t0:.1f}s total)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
