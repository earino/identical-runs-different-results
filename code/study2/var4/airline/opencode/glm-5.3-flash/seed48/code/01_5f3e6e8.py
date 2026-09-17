"""XGBoost binary classifier: airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _tod_minutes(s: pd.Series) -> pd.Series:
    t = pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")
    return (t // 100) * 60 + (t % 100)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    ang = 2 * np.pi * _tod_minutes(df["DepTime"]) / 1440.0
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["tod_min"] = _tod_minutes(df["DepTime"])
    X["month"] = pd.to_numeric(df["Month"].str[2:], errors="coerce").fillna(0).astype("int64")
    X["day"] = pd.to_numeric(df["DayofMonth"].str[2:], errors="coerce").fillna(0).astype("int64")
    X["dow"] = pd.to_numeric(df["DayOfWeek"].str[2:], errors="coerce").fillna(0).astype("int64")
    X["distance"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce").fillna(0))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
tr_idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(tr_idx)
cut = int(0.85 * len(tr_idx))
fit_idx, val_idx = tr_idx[:cut], tr_idx[cut:]
model.fit(
    prepare(train.iloc[fit_idx]),
    to_y(train.iloc[fit_idx]),
    eval_set=[(prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx]))],
    verbose=0,
)
best_ntree = model.best_iteration + 1
print(f"Early-stopped at {best_ntree} trees, val AUC {model.best_score:.4f}, time {time.time() - t0:.1f}s")
# refit on full training data with the chosen number of trees
model = xgb.XGBClassifier(
    n_estimators=best_ntree,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(prepare(train), to_y(train))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
