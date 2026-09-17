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
CONFIGS = [
    (d, s)
    for d in (4, 5, 6, 7, 8)
    for s in (0.6, 0.7, 0.8, 0.9, 1.0)
]


def make_model(seed: int, depth: int = 6, sub: float = 0.8) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=depth,
        subsample=sub,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
y_train = to_y(train)
X_train = prepare(train)
models = []
for i, (depth, sub) in enumerate(CONFIGS):
    m = make_model(SEED + i, depth, sub)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
