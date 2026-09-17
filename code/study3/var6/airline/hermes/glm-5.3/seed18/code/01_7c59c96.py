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


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _cnum(s: pd.Series) -> pd.Series:
    """c-<n> string column -> numeric n."""
    return _num(s.astype(str).str.replace("c-", "", regex=False))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _cnum(df["Month"])
    X["DayofMonth"] = _cnum(df["DayofMonth"])
    X["DayOfWeek"] = _cnum(df["DayOfWeek"])
    dep = _num(df["DepTime"])
    X["DepTime"] = dep
    X["Distance"] = _num(df["Distance"])
    X["DepHour"] = dep // 100
    X["DepMinute"] = dep % 100
    X["MinuteOfDay"] = X["DepHour"] * 60 + X["DepMinute"]
    X["sin_hour"] = np.sin(2 * np.pi * X["MinuteOfDay"] / 1440)
    X["cos_hour"] = np.cos(2 * np.pi * X["MinuteOfDay"] / 1440)
    X["sin_month"] = np.sin(2 * np.pi * X["Month"] / 12)
    X["cos_month"] = np.cos(2 * np.pi * X["Month"] / 12)
    X["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    X["Route"] = X["Route"].astype(str)
    for c in CAT_COLS + ["Route"]:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    return X


_src = train.copy()
_src["Route"] = _src["Origin"].astype(str) + "_" + _src["Dest"].astype(str)
CAT_LEVELS = {c: pd.Index(sorted(_src[c].dropna().unique())) for c in CAT_COLS + ["Route"]}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr = prepare(train)
Xev = prepare(evald)
model.fit(Xtr, to_y(train), eval_set=[(Xev, to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iter: {model.best_iteration}")
print(f"n_features: {Xtr.shape[1]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, model.best_iteration + 1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
