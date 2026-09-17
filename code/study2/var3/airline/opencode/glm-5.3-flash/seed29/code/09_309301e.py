"""Experiment 19: block granularity sweep (5/10/15-min, multi-scale).

Exp18: 15-min blocks (96) -> 0.7233 > 30-min (0.7223) > dow_block (0.7207); freq neutral.
Finer blocks isolate sharper time-of-day effects for shallow trees. Try 5-min (288),
10-min (144), and 15-min + coarse 60-min multi-scale.
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


def _cat(series: pd.Series, n_levels: int, prefix: str = "") -> pd.Categorical:
    return pd.Categorical(series.astype(str), categories=[prefix + str(i) for i in range(n_levels)])


def prepare(df: pd.DataFrame, variant: str = "D") -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_missing"] = dt.isna().astype(float)
    dtf = dt.fillna(0)
    h = (dtf // 100).astype(float) % 24
    total_min = h * 60 + dtf % 100
    X["dep_min"] = total_min
    if variant == "G":
        X["dep_block"] = _cat((total_min // 5).astype(int), 288)
    elif variant == "H":
        X["dep_block"] = _cat((total_min // 10).astype(int), 144)
    elif variant == "J":
        X["dep_block"] = _cat((total_min // 15).astype(int), 96)
        X["dep_hour"] = _cat(h.astype(int), 24)
    else:
        X["dep_block"] = _cat((total_min // 15).astype(int), 96)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
    month = _ord(df["Month"]).fillna(1).clip(1, 12).astype(int)
    X["dayofyear"] = CUM_DAYS[month - 1] + _ord(df["DayofMonth"]).fillna(15)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


MEMBERS = [(2, 250), (2, 500), (3, 250), (3, 500), (4, 120), (4, 250), (4, 500), (5, 80)]

t0 = time.time()
yfull = to_y(train)
yev = to_y(evald)

results = {}
for variant in ["D", "G", "H", "J"]:
    Xfull = prepare(train, variant)
    Xev = prepare(evald, variant)
    preds = []
    for d, n in MEMBERS:
        m = xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        )
        m.fit(Xfull, yfull)
        preds.append(m.predict_proba(Xev)[:, 1])
    ea = roc_auc_score(yev, np.mean(preds, axis=0))
    results[variant] = ea
    print(f"variant={variant}: eval={ea:.4f}")
print(f"variants took {time.time() - t0:.1f}s")

best = max(results, key=results.get)
print(f"best variant: {best}")

_models = []


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df, best)
    return np.mean([m.predict_proba(X)[:, 1] for m in _models], axis=0)


Xfull = prepare(train, best)
for d, n in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xfull, yfull)
    _models.append(m)

eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")