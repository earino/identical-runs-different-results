"""XGBoost binary classifier for airline delays. Only file the agent edits.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
# interaction categoricals: entity x scheduled-departure-hour, day-of-week x hour
INTER_PAIRS = [
    ("UniqueCarrier", "hour"),
    ("Origin", "hour"),
    ("Dest", "hour"),
    ("DayOfWeek", "hour"),
]


# --- features -----------------------------------------------------------------
def _fe(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row feature engineering (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    dt = X["DepTime"]
    X["hour"] = (dt // 100) % 24
    X["minute"] = dt % 100
    X["Distance"] = df["Distance"].astype(float)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayofMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    return X


CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
_TRAIN_PREP = _fe(train)
INTER_LEVELS = {
    (_a, _b): pd.Index(
        sorted((train[_a].astype(str) + "_" + _TRAIN_PREP[_b].astype(str)).unique())
    )
    for _a, _b in INTER_PAIRS
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    X = _fe(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    for a, b in INTER_PAIRS:
        key = (df[a].astype(str) + "_" + X[b].astype(str)).astype(str)
        X[f"{a}__{b}"] = pd.Categorical(key, categories=INTER_LEVELS[(a, b)])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
_BASE_PARAMS = dict(
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
# five members spread across the regularization spectrum (two alpha levels repeated with a
# different seed): averaging decorrelated regularized trees lifts the blend above any member
MEMBERS = [
    dict(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=20, colsample_bytree=0.7, reg_alpha=0.3),
    dict(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=20, colsample_bytree=0.7, reg_alpha=3.0, gamma=1.0),
    dict(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=20, colsample_bytree=0.7, reg_alpha=10.0),
    dict(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=20, colsample_bytree=0.7, reg_alpha=0.3, random_state=43),
    dict(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=20, colsample_bytree=0.7, reg_alpha=15.0),
]

t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
models = []
for spec in MEMBERS:
    m = xgb.XGBClassifier(**{**_BASE_PARAMS, **spec})
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
