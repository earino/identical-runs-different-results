"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design: time-of-day features dominate and are year-stable; high-cardinality route/TE features and big
deep models do not transfer 2005->2006. Small ensemble of shallow XGBoost models (varied colsample
seeds), with late-2005 rows upweighted (recency helps the time-shifted holdout).
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
N_BAG = 10
# depth-diverse bag: averaging over tree depths beats a single depth (variance reduction)
BAG_DEPTHS = [2, 3, 4, 5, 6] * 2

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering ---------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here; predict_proba() calls it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = pd.to_numeric(df[c].astype(str).str.replace("c-", "", regex=False), errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24)
    X["DepHour"] = hour
    X["DepMinutes"] = hour * 60 + (dep % 100).clip(0, 59)
    X["MinuteOfHour"] = (dep % 100).clip(0, 59)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- training --------------------------------------------------------------------
y = to_y(train)
# recency weighting: the eval/holdout year follows late 2005, so upweight Oct-Dec 2005
month = pd.to_numeric(train["Month"].astype(str).str.replace("c-", "", regex=False)).to_numpy()
w = np.where(month >= 10, 2.0, 1.0)

t0 = time.time()
Xtr = prepare(train)
models = []
for s in range(N_BAG):
    m = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=BAG_DEPTHS[s],
        learning_rate=0.05,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=1000 + s,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, y, sample_weight=w)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
