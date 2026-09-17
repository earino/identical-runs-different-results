"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature definition ---------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["Distance"]
# category levels fitted on training data only; unseen levels at predict time become NaN
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _cyc(v, period):
    a = 2.0 * np.pi * v / period
    return np.sin(a), np.cos(a)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # calendar: "c-<n>" strings -> integers
    month = df["Month"].str.slice(2).astype(int)
    dom = df["DayofMonth"].str.slice(2).astype(int)
    dow = df["DayOfWeek"].str.slice(2).astype(int)
    X["Month"] = month
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    ms, mc = _cyc(month - 1, 12.0)
    X["month_sin"], X["month_cos"] = ms, mc
    ds, dc = _cyc(dom - 1, 31.0)
    X["dom_sin"], X["dom_cos"] = ds, dc
    ws, wc = _cyc(dow - 1, 7.0)
    X["dow_sin"], X["dow_cos"] = ws, wc
    # scheduled departure time (hhmm; values >= 2400 roll into the next day)
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    tod = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_tod"] = tod
    ts, tc = _cyc(tod, 1440.0)
    X["tod_sin"], X["tod_cos"] = ts, tc
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    cat_levels = CAT_LEVELS
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=7,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=25,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

Xtr = prepare(train)
ytr = to_y(train)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(Xtr))
n_val = len(Xtr) // 10
val_idx, fit_idx = idx[:n_val], idx[n_val:]

t0 = time.time()
model.fit(
    Xtr.iloc[fit_idx],
    ytr[fit_idx],
    eval_set=[(Xtr.iloc[val_idx], ytr[val_idx])],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iteration={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
