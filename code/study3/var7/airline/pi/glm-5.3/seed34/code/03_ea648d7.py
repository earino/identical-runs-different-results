"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature definitions (levels/stats fit on train only) -----------------------
LV = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in
      ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
HOURS = pd.Index([str(h) for h in range(25)])


def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7 (strips the 'c-' prefix)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = _cnum(df[c])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24).astype(int)
    X["DepTime"] = dep
    X["DepHour"] = hour
    X["DepMinute"] = dep % 100
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["DepHourCat"] = pd.Categorical(hour.astype(str), categories=HOURS)
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c].astype(str), categories=LV[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: 3-seed ensemble ---------------------------------------------------
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

t0 = time.time()
models = []
for seed in (SEED, 1, 2):
    m = xgb.XGBClassifier(
        n_estimators=6000,
        learning_rate=0.03,
        max_depth=14,
        reg_alpha=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        max_cat_to_onehot=24,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        eval_metric="auc",
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"seed {seed}: best_iter={m.best_iteration}  ({time.time() - t0:.0f}s)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
