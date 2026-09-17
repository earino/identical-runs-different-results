"""XGBoost binary classifier on airline delays. Only file the agent edits.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "route"]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering. Called on train (module level, for fitting) and inside predict_proba."""
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(np.int32)
    dt = dt.where(dt < 2400, dt - 2400)          # 2400..2411 -> 0..11 (past midnight)
    hour = dt // 100
    minute = dt % 100
    tod = hour * 60 + minute                     # minutes since midnight

    X["deptime"] = dt
    X["hour"] = hour
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["distance"] = df["Distance"].astype(np.float32)
    X["dist_log"] = np.log1p(df["Distance"].astype(np.float32))
    X["route"] = df["Origin"].astype(str) + ">" + df["Dest"].astype(str)
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c].astype(str)
    return X


# fit category levels on training data only
_fit = _engineer(train)
cat_levels = {c: pd.Index(sorted(_fit[c].unique())) for c in CAT_COLS}
FEATURE_NAMES = [c for c in _fit.columns if c not in CAT_COLS] + CAT_COLS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = _engineer(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[FEATURE_NAMES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr, X_val, y_tr, y_val = train_test_split(
    prepare(train), to_y(train), test_size=0.2, random_state=SEED, stratify=to_y(train))
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
