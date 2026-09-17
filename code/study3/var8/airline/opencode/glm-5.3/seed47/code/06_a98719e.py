"""XGBoost airline delay. Contract: prints `Eval AUC: 0.xxxx`; predict_proba(df) works on raw rows."""
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

rng = np.random.RandomState(SEED)
val_idx = rng.choice(len(train), size=10000, replace=False)
mask = np.ones(len(train), dtype=bool)
mask[val_idx] = False

# --- fitted on train only -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _c_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _c_num(df["Month"])
    X["DayofMonth"] = _c_num(df["DayofMonth"])
    X["DayOfWeek"] = _c_num(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) % 2400
    X["DepTime"] = dep
    X["Hour"] = (dep // 100).astype(int)
    X["MinOfDay"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ALL = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
       "LogDist", "UniqueCarrier", "Origin", "Dest"]

Xtr, Xev = prepare(train), prepare(evald)
ytr, yv, yev = to_y(train[mask]), to_y(train[~mask]), to_y(evald)


def bag(params, seeds=(1, 2, 3, 4, 5)):
    preds, models = [], []
    for s in seeds:
        base = dict(n_estimators=3000, learning_rate=0.05, tree_method="hist", enable_categorical=True,
                    eval_metric="auc", early_stopping_rounds=30, subsample=0.9, colsample_bytree=0.9,
                    random_state=s, n_jobs=N_JOBS)
        base.update(params)
        m = xgb.XGBClassifier(**base)
        m.fit(Xtr[mask][ALL], ytr, eval_set=[(Xtr[~mask][ALL], yv)], verbose=False)
        preds.append(m.predict_proba(Xev[ALL])[:, 1])
        models.append(m)
    ev = roc_auc_score(yev, np.mean(preds, axis=0))
    va = roc_auc_score(yv, np.mean([m.predict_proba(Xtr[~mask][ALL])[:, 1] for m in models], axis=0))
    iters = ",".join(str(m.best_iteration) for m in models)
    print(f"  bag{len(seeds)} iters={iters} valid={va:.4f} eval={ev:.4f}")
    return models, ev


CONFIGS = [
    ("bag5_d4_mcw50", dict(max_depth=4, min_child_weight=50)),
    ("bag5_d4_mcw50_sub07", dict(max_depth=4, min_child_weight=50, subsample=0.7, colsample_bytree=0.7)),
    ("bag5_d3_mcw50", dict(max_depth=3, min_child_weight=50)),
]
best = None
for name, params in CONFIGS:
    t0 = time.time()
    print(f"[{name}]")
    models, ev = bag(params)
    print(f"  total t={time.time()-t0:.1f}s")
    if best is None or ev > best[2]:
        best = (name, models, ev)

# single-model reference
mref = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=50,
                         tree_method="hist", enable_categorical=True, eval_metric="auc",
                         early_stopping_rounds=30, subsample=0.9, colsample_bytree=0.9,
                         random_state=SEED, n_jobs=N_JOBS)
mref.fit(Xtr[mask][ALL], ytr, eval_set=[(Xtr[~mask][ALL], yv)], verbose=False)
print(f"[single_ref] eval={roc_auc_score(yev, mref.predict_proba(Xev[ALL])[:, 1]):.4f}")

name, MODELS, eval_auc = best
print(f"best: {name}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[ALL]
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


print(f"Eval AUC: {roc_auc_score(yev, predict_proba(evald)):.4f}")
