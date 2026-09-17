"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
NUM_COLS = ["Month", "DayofMonth", "DayOfWeek"]  # stored as c-<n> strings
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
cat_levels = {}


def _strip_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in NUM_COLS:
        X[c] = _strip_c(df[c])
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    tod = hour * 60 + minute  # minutes since midnight
    X["DepHour"] = hour
    X["TodMin"] = tod
    X["TodSin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["TodCos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in ["UniqueCarrier", "Origin", "Dest", "Route"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# fit category levels on training data only
_tmp = train.copy()
_tmp["Route"] = _tmp["Origin"].astype(str) + "_" + _tmp["Dest"].astype(str)
for c in ["UniqueCarrier", "Origin", "Dest", "Route"]:
    cat_levels[c] = pd.Index(sorted(_tmp[c].dropna().unique()))

# --- model --------------------------------------------------------------------
# internal validation split from train (2005) for early stopping
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_tr = int(0.9 * len(train))
tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=40,
)

t0 = time.time()
Xp = prepare(train)
model.fit(Xp.iloc[tr_idx], to_y(train)[tr_idx], eval_set=[(Xp.iloc[va_idx], to_y(train)[va_idx])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iters: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
