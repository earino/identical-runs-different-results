"""Experiment 21: semantic bagging + week-of-year categorical.

Ensemble mix is saturated (M2 0.7235 ~ M1 0.7234). New diversity axes:
  V1 = M2 reference
  V2 = semantic bagging: members trained with different feature groups dropped
  V3 = base + week-of-year categorical (53 levels; holiday weeks drive delays)
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


def prepare(df: pd.DataFrame, variant: str = "V1", drop: str = "") -> pd.DataFrame:
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
    dom = _ord(df["DayofMonth"]).fillna(15)
    X["dayofyear"] = CUM_DAYS[month - 1] + dom
    if variant == "V3":
        X["weekofyear"] = _cat(((X["dayofyear"] - 1) // 7).astype(int), 53)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    if drop:
        X = X.drop(columns=[c for c in X.columns if c.startswith(drop)])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


MEMBERS = [(2, 250), (2, 500), (3, 250), (3, 500), (4, 120), (4, 250), (4, 500), (5, 80), (5, 150), (6, 80)]
DROPS = ["", "dayofyear", "Distance", "Origin", "Dest", "dep_min", "Month", "DayofMonth"]

t0 = time.time()
yfull = to_y(train)
yev = to_y(evald)

results = {}


def run_ensemble(tag, variant, drops):
    preds = []
    for i, (d, n) in enumerate(MEMBERS):
        drop = drops[i % len(drops)]
        Xtr_f = prepare(train, variant, drop)
        m = xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED + i, n_jobs=N_JOBS,
        )
        m.fit(Xtr_f, yfull)
        preds.append(m.predict_proba(prepare(evald, variant, drop))[:, 1])
    ea = roc_auc_score(yev, np.mean(preds, axis=0))
    results[tag] = ea
    print(f"{tag}: eval={ea:.4f}")


run_ensemble("V1_ref", "V1", [""])
run_ensemble("V2_sembag", "V1", DROPS)
run_ensemble("V3_woy", "V3", [""])
print(f"variants took {time.time() - t0:.1f}s")

best = max(results, key=results.get)
print(f"best variant: {best}")

_models = []


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    if best == "V2_sembag":
        preds = [m.predict_proba(prepare(df, "V1", drop))[:, 1] for m, drop in _models]
        return np.mean(preds, axis=0)
    X = prepare(df, best)
    return np.mean([m.predict_proba(X)[:, 1] for m in _models], axis=0)


if best == "V2_sembag":
    for i, (d, n) in enumerate(MEMBERS):
        drop = DROPS[i % len(DROPS)]
        m = xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED + i, n_jobs=N_JOBS,
        )
        m.fit(prepare(train, "V1", drop), yfull)
        _models.append((m, drop))
else:
    for d, n in MEMBERS:
        m = xgb.XGBClassifier(
            n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
            enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        )
        m.fit(prepare(train, best), yfull)
        _models.append(m)

eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")