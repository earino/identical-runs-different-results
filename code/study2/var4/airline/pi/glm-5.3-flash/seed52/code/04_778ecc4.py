"""XGBoost binary classifier for airline delay. ONLY FILE THE AGENT EDITS.

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
N_MODELS = 5

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"]
    X["DayofMonth"] = df["DayofMonth"]
    X["DayOfWeek"] = df["DayOfWeek"]
    X["DepTime"] = df["DepTime"]
    X["DepHour"] = df["DepTime"] // 100
    X["DepMinute"] = df["DepTime"] % 100
    X["Distance"] = df["Distance"]
    X["LogDistance"] = np.log1p(df["Distance"])
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["Route"] = df["Origin"] + "_" + df["Dest"]
    return X


train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# category levels fitted on TRAINING data only; unseen levels in later data -> NaN
_base_train = base_features(train)
cat_levels = {c: pd.Index(sorted(_base_train[c].dropna().unique())) for c in CAT_COLS}
del _base_train


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of L1-regularized XGBoost models, early-stopped on eval (2006 proxy) ---
PARAMS = dict(
    n_estimators=3000,
    max_depth=5,
    learning_rate=0.1,
    colsample_bytree=0.7,
    reg_alpha=8.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=100,
    eval_metric="auc",
)

t0 = time.time()
X_train, X_eval = prepare(train), prepare(evald)
y_train, y_eval = to_y(train), to_y(evald)
models = []
for s in range(N_MODELS):
    m = xgb.XGBClassifier(**{**PARAMS, "random_state": SEED + s})
    m.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Iterations: {[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
