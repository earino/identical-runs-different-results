"""XGBoost binary classifier (airline delays). Only file the agent edits.

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

BASE_FEATS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "DepHour"]
HOURS = list(range(24))

# encoders fitted on training data only (module level = fitted once, used by prepare for any df)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS if c in train.columns}


def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row engineered features. No statistics from other rows."""
    out = df[BASE_FEATS].copy()
    dt = df["DepTime"].astype(np.int64)
    hh = dt // 100
    mm = dt % 100
    # DepTime >= 2400 rolls past midnight (e.g. 2620 = 02:20 next day)
    mins = (hh * 60 + mm) % 1440
    out["DepTimeMin"] = mins
    hour = (hh % 24).to_numpy()
    out["DepHour"] = pd.Categorical(hour, categories=HOURS)
    mon = df["Month"].str[2:].astype(int).to_numpy()
    dom = df["DayofMonth"].str[2:].astype(int).to_numpy()
    dow = df["DayOfWeek"].str[2:].astype(int).to_numpy()
    out["sinM"] = np.sin(2 * np.pi * mon / 12)
    out["cosM"] = np.cos(2 * np.pi * mon / 12)
    out["sinD"] = np.sin(2 * np.pi * dom / 31)
    out["cosD"] = np.cos(2 * np.pi * dom / 31)
    out["sinW"] = np.sin(2 * np.pi * dow / 7)
    out["cosW"] = np.cos(2 * np.pi * dow / 7)
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = add_feats(df)
    for c in CAT_COLS:
        if c in cat_levels:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
params = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.5,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=100,
    verbosity=0,
)

t0 = time.time()
model = xgb.XGBClassifier(**params)
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
model.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
best_it = model.best_iteration
eval_auc = roc_auc_score(yev, model.predict_proba(Xev)[:, 1])
print(f"Early-stop fit: best_iteration={best_it} auc={eval_auc:.4f} ({time.time() - t0:.1f}s)")

# refit on training data only with the chosen number of trees
t0 = time.time()
final_params = dict(params)
final_params["n_estimators"] = best_it + 1
final_params["early_stopping_rounds"] = None
model = xgb.XGBClassifier(**final_params)
model.fit(Xtr, ytr, verbose=False)
print(f"Refit time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
