"""XGBoost binary classifier on the airline delay task.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# fitted on training data only
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# recency weighting: later months of 2005 count more (bridge toward 2006)
month_num = train["Month"].str[2:].astype(int).to_numpy()
SAMPLE_W = 1.0 + 1.5 * (month_num - 1) / 11.0

# members: (max_depth, subsample, colsample_bytree, learning_rate, n_estimators, seed)
MEMBERS = [
    (26, 0.75, 0.5, 0.05, 109, 999),
    (24, 0.8, 0.5, 0.05, 110, 7),
    (26, 0.7, 0.5, 0.05, 116, 123),
    (24, 0.9, 0.6, 0.03, 130, 500),
    (20, 0.7, 0.4, 0.08, 90, 321),
    (16, 0.75, 0.5, 0.07, 110, 888),
    (30, 0.7, 0.5, 0.05, 110, 777),
]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayofMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    X["DepTime"] = df["DepTime"].astype(int)
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["Distance"] = df["Distance"].astype(float)
    X["logDist"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
models = []
for depth, ss, cs, lr, n_est, seed in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=depth,
        subsample=ss,
        colsample_bytree=cs,
        tree_method="hist",
        max_bin=512,
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X_all, y_all, sample_weight=SAMPLE_W)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)  # prepare once, reuse across members
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
