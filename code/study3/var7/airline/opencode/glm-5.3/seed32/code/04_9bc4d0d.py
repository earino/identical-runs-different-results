"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); fitted statistics come from train.csv only.

Regime discovered so far: 2005->2006 distribution shift punishes capacity and route-level
memorization; robustness comes from deep trees + row/col subsampling + averaging over seeds.
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

# --- features -----------------------------------------------------------------
CINT_COLS = ["DayOfWeek"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]

_raw = None  # filled below, fitted on train only


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CINT_COLS:
        X[c] = pd.to_numeric(df[c].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    t = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    hh = (t // 100) % 24
    X["Hour"] = hh
    X["MinuteOfDay"] = hh * 60 + t % 100
    X["DepTime"] = t
    a = 2 * np.pi * X["MinuteOfDay"] / 1440.0
    X["SinT"] = np.sin(a)
    X["CosT"] = np.cos(a)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDistance"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    return X


CAT_LEVELS = {c: pd.Index(sorted(base_features(train)[c].unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of deep subsampled XGB models ----------------------
X_all = prepare(train)
y_all = to_y(train)

PARAMS = dict(
    n_estimators=25,
    max_depth=28,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=0.1,
    colsample_bytree=0.5,
    subsample=0.8,
    max_bin=512,
    n_jobs=N_JOBS,
)
# diverse near-tied configs (n_estimators, max_depth, max_bin, colsample, subsample)
CONFIGS = [
    dict(),
    dict(max_depth=32),
    dict(n_estimators=30, max_bin=1024),
    dict(subsample=1.0),
    dict(n_estimators=40, max_depth=24, max_bin=256),
]
N_SEEDS = 2  # per config -> 10 members total

t0 = time.time()
models = []
for i, cfg in enumerate(CONFIGS):
    for s in range(N_SEEDS):
        m = xgb.XGBClassifier(random_state=SEED + (i * N_SEEDS + s) * 101, **{**PARAMS, **cfg})
        m.fit(X_all, y_all)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = np.zeros(len(X))
    for m in models:
        preds += m.predict_proba(X)[:, 1]
    return preds / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
