"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    # c-<n> strings -> ordered integers
    X["Month"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    X["DayofMonth"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    X["Hour"] = dep // 100
    X["Minute"] = dep % 100
    X["Distance"] = df["Distance"].astype(float)
    X["log_Distance"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=12,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
rng = np.random.RandomState(SEED)
tr_idx = rng.rand(len(train)) < 0.9
Xtr, ytr = prepare(train[tr_idx]), to_y(train[tr_idx])
Xva, yva = prepare(train[~tr_idx]), to_y(train[~tr_idx])
model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
n_best = model.best_iteration + 1
print(f"Training time: {time.time() - t0:.1f}s, best iter: {model.best_iteration}")
# stage 2: refit on ALL training rows with the tuned iteration count
t1 = time.time()
model.set_params(n_estimators=int(n_best), early_stopping_rounds=None)
model.fit(prepare(train), to_y(train), verbose=False)
print(f"Refit time: {time.time() - t1:.1f}s on {int(n_best)} trees")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
