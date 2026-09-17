"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature engineering (fitted on TRAIN only, applied inside prepare) --------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
y_train = (train[TARGET] == POSITIVE).astype(int)
GLOBAL_MEAN = float(y_train.mean())


def _add_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row features (no fitted state)."""
    X = df[["DepTime", "Distance"] + CAT_COLS].copy()
    dt = df["DepTime"].astype(int)
    X["hour"] = dt // 100
    X["minute"] = dt % 100
    X["depm_frac"] = (X["hour"] * 60 + X["minute"]) / 1440.0
    X["redeye"] = ((dt < 600) | (dt >= 2100)).astype(int)
    X["month_n"] = df["Month"].str[2:].astype(int)
    X["dow_n"] = df["DayOfWeek"].str[2:].astype(int)
    X["dom_n"] = df["DayofMonth"].str[2:].astype(int)
    return X


# smoothed target encodings, fitted on train only (DISABLED: overfit 2005 -> eval drop, exp 3)
TE_ORIGIN = TE_DEST = TE_CARRIER = TE_ROUTE = TE_O_H3 = TE_D_H3 = None


def _h3(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(int) // 300).astype(str)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _add_basic(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

PARAMS = dict(
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
)

t0 = time.time()
es_model = xgb.XGBClassifier(
    n_estimators=3000,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=50,
    **PARAMS,
)
es_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
best_n = int(es_model.best_iteration) + 1
print(f"ES fit: {time.time() - t0:.1f}s, best_iteration={best_n}, va_auc={es_model.best_score:.4f}")

model = xgb.XGBClassifier(
    n_estimators=best_n,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    **PARAMS,
)

t0 = time.time()
model.fit(X, y)
print(f"Final fit ({best_n} trees): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
