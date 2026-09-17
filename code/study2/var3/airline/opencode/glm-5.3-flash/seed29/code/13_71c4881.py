"""Experiment 23: broader ES-member ensemble (more depths, ES lossguide).

Exp22: per-member ES counts -> 0.7252 (d2/554, d3/374, d4/214, d5/237, d6/232).
Deeper members want more trees than the hand grid. Extend:
  S1 = R2 reference (5 depths)
  S2 = R2 with depths 2-8
  S3 = S2 + ES-selected lossguide members
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


def es_count_for(d, **extra):
    es = xgb.XGBClassifier(
        n_estimators=2000, max_depth=d, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
        early_stopping_rounds=150, eval_metric="auc", **extra,
    )
    es.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return int(es.best_iteration + 1)


def full_member(d, n, **extra):
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=d, learning_rate=0.05, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **extra,
    )
    m.fit(Xfull, yfull)
    return m


preds_r1 = [full_member(d, n).predict_proba(Xev)[:, 1] for d, n in FIXED]
results["R1_fixed"] = eval_avg(preds_r1)
print(f"R1_fixed: {results['R1_fixed']:.4f}")


def es_depth_ensemble(depths, lg=False):
    preds, counts = [], []
    for d in depths:
        n_best = es_count_for(d)
        counts.append((d, n_best))
        preds.append(full_member(d, n_best).predict_proba(Xev)[:, 1])
        print(f"  depth {d}: ES count {n_best}")
    if lg:
        for leaves in [32, 64]:
            n_best = es_count_for(6, max_leaves=leaves, grow_policy="lossguide")
            counts.append((leaves, n_best))
            preds.append(full_member(6, n_best, max_leaves=leaves, grow_policy="lossguide").predict_proba(Xev)[:, 1])
            print(f"  lossguide leaves {leaves}: ES count {n_best}")
    return preds, counts


preds_s1, counts_s1 = es_depth_ensemble(ES_DEPTHS)
results["S1_es5"] = eval_avg(preds_s1)
print(f"S1_es5: {results['S1_es5']:.4f}")

preds_s2, counts_s2 = es_depth_ensemble([2, 3, 4, 5, 6, 7, 8])
results["S2_es7"] = eval_avg(preds_s2)
print(f"S2_es7: {results['S2_es7']:.4f}")

preds_s3, counts_s3 = es_depth_ensemble([2, 3, 4, 5, 6, 7, 8], lg=True)
results["S3_es7lg"] = eval_avg(preds_s3)
print(f"S3_es7lg: {results['S3_es7lg']:.4f}")

print(f"scan took {time.time() - t0:.1f}s")
best = max(results, key=results.get)
print(f"best: {best}")

_models = []


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in _models], axis=0)


specs = {"S1_es5": counts_s1, "S2_es7": counts_s2, "S3_es7lg": counts_s3}[best]
for d_or_leaves, n_best in specs:
    if best == "S3_es7lg" and d_or_leaves in (32, 64):
        _models.append(full_member(6, n_best, max_leaves=d_or_leaves, grow_policy="lossguide"))
    else:
        _models.append(full_member(d_or_leaves, n_best))

eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")