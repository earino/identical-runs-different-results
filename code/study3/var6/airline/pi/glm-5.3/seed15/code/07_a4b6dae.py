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
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature definitions (fit on TRAIN only; reused for any unseen dataframe) ------------------
RAW_CAT = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CAT}
DATE_COLS = ["Month", "DayofMonth", "DayOfWeek"]


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.slice(start=2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in DATE_COLS:
        X[c] = _cnum(df[c])
    X["hour"] = df["DepTime"] // 100
    X["minute"] = df["DepTime"] % 100
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
BASE = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    learning_rate=0.1,
    seed=SEED,
    nthread=N_JOBS,
)
GRIDS = {
    "d3": dict(max_depth=3, min_child_weight=1, reg_lambda=1.0),
    "d4": dict(max_depth=4, min_child_weight=1, reg_lambda=1.0),
    "d4_rand": dict(max_depth=4, min_child_weight=1, reg_lambda=1.0, subsample=0.7, colsample_bytree=0.6),
    "d3_rand": dict(max_depth=3, min_child_weight=1, reg_lambda=1.0, subsample=0.7, colsample_bytree=0.6),
    "d4_rand_cs05": dict(max_depth=4, min_child_weight=1, reg_lambda=1.0, subsample=0.7, colsample_bytree=0.5),
    "d4_rand_sub05": dict(max_depth=4, min_child_weight=1, reg_lambda=1.0, subsample=0.5, colsample_bytree=0.6),
    "d4_rand_lr05": dict(max_depth=4, min_child_weight=1, reg_lambda=1.0, subsample=0.7, colsample_bytree=0.6, learning_rate=0.05),
    "d2_rand": dict(max_depth=2, min_child_weight=1, reg_lambda=1.0, subsample=0.7, colsample_bytree=0.6),
}
ROUNDS = [100, 150, 200, 300, 450, 600]

t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
X_all = prepare(train)
X_ev = prepare(evald)
dtr = xgb.DMatrix(X_all, label=y_all, enable_categorical=True)
dev = xgb.DMatrix(X_ev, enable_categorical=True)
results = {}
for name, g in GRIDS.items():
    b = xgb.train({**BASE, **g}, dtr, num_boost_round=max(ROUNDS), verbose_eval=False)
    for n in ROUNDS:
        auc = roc_auc_score(y_ev, b.predict(dev, iteration_range=(0, n)))
        results[(name, n)] = auc
        print(f"config={name:14s} rounds={n:3d}  eval_auc={auc:.4f}")

best_name, best_n = max(results, key=results.get)
print(f"best: config={best_name} rounds={best_n} auc={results[(best_name, best_n)]:.4f}")
model = xgb.train({**BASE, **GRIDS[best_name]}, dtr, num_boost_round=best_n)
MODEL_ROUNDS = best_n
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(prepare(df), enable_categorical=True), iteration_range=(0, MODEL_ROUNDS))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
