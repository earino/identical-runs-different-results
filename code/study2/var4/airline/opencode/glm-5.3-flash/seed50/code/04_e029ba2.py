"""XGBoost binary classifier on airline delays. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique()))
              for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
HOURS = pd.Index([f"h{h:02d}" for h in range(24)])
DIST_BINS = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2500, 6000]


def _dep_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).where(dt < 2400)
    minute = dt.where(dt < 2400) % 100
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    out = pd.DataFrame({
        "DepHour": ("h" + hour.astype("Int64").astype(str)).fillna("na"),
        "DepMinute": minute,
        "DepTimeSin": np.sin(2 * np.pi * (hour * 60 + minute) / 1440.0),
        "DepTimeCos": np.cos(2 * np.pi * (hour * 60 + minute) / 1440.0),
        "DistanceLog": np.log1p(dist),
        "DistBin": pd.Categorical(pd.cut(dist, DIST_BINS).astype(str)),
    })
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]].copy()
    for c in X.columns:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    derived = _dep_time_feats(df)
    derived["DepHour"] = pd.Categorical(derived["DepHour"], categories=HOURS)
    out = pd.concat([X, derived], axis=1)
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_full, y_full = prepare(train), to_y(train)
X_tr, X_va, y_tr, y_va = train_test_split(X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full)
finder = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=50, eval_metric="auc", **PARAMS)
finder.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_n = int(getattr(finder, "best_iteration", 0)) + 1
print(f"Best rounds: {best_n} (val auc {finder.best_score:.4f})")
MEMBERS = [
    dict(max_depth=8, random_state=SEED),
    dict(max_depth=8, random_state=7),
    dict(max_depth=7, random_state=3, min_child_weight=10),
    dict(max_depth=9, random_state=11, subsample=0.8, colsample_bytree=0.8),
    dict(max_depth=6, random_state=13, learning_rate=0.06),
    dict(max_depth=7, random_state=23, subsample=0.8, colsample_bynode=0.8),
]
models = []
for i, over in enumerate(MEMBERS):
    p = dict(PARAMS)
    p.update(over)
    m = xgb.XGBClassifier(n_estimators=best_n, **p)
    m.fit(X_full, y_full)
    models.append(m)
    print(f"member {i} done")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
