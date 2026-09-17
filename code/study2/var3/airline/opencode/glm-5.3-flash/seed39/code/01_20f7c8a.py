"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["Month", "UniqueCarrier", "Origin", "Dest"]
CATS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(np.int64)
    X["dep_min"] = (dt // 100) * 60 + dt % 100
    X["dep_hour"] = dt // 100
    m = X["dep_min"] % 1440
    X["dep_sin"] = np.sin(2 * np.pi * m / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * m / 1440)
    X["month"] = df["Month"].astype(str).str[2:].astype(np.int64)
    X["dom"] = df["DayofMonth"].astype(str).str[2:].astype(np.int64)
    X["dow"] = df["DayOfWeek"].astype(str).str[2:].astype(np.int64)
    X["distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CATS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=6,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=30,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr, X_val, y_tr, y_val = train_test_split(
    train, to_y(train), test_size=0.2, random_state=SEED, shuffle=True
)
model.fit(prepare(X_tr), y_tr, eval_set=[(prepare(X_val), y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s (trees={model.best_iteration})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
