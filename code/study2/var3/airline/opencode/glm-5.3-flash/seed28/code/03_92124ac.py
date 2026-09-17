"""XGBoost binary classifier on the airline dataset. Only file the agent edits.

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

# --- feature engineering ------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
train_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
cat_levels["Route"] = pd.Index(sorted(train_route.unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything
    # computed on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = pd.to_numeric(df["Month"].str.slice(2), errors="coerce")
    X["DayofMonth"] = pd.to_numeric(df["DayofMonth"].str.slice(2), errors="coerce")
    X["DayOfWeek"] = pd.to_numeric(df["DayOfWeek"].str.slice(2), errors="coerce")
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = (dt // 100).clip(0, 23)
    mm = (dt % 100).clip(0, 59)
    dep_min = hh * 60 + mm
    X["DepTime"] = dep_min
    ang = 2 * np.pi * dep_min / 1440.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    X["Hour"] = hh
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c]
    X["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in ["UniqueCarrier", "Origin", "Dest", "Route"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)


def fit_one(seed: int, cap: int) -> xgb.XGBClassifier:
    Xtr, Xva, ytr, yva = train_test_split(
        X_all, y_all, test_size=0.1, random_state=seed, stratify=y_all
    )
    m = xgb.XGBClassifier(
        n_estimators=cap,
        learning_rate=0.06,
        max_depth=8,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=60,
        eval_metric="auc",
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"seed={seed} best_iter={m.best_iteration}")
    return m


t0 = time.time()
CAPS = [800, 1000, 1200, 1400, 1600]
models = [fit_one(s, c) for s, c in zip([42, 7, 123, 2024, 555], CAPS)]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    return np.mean([m.predict_proba(P)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
