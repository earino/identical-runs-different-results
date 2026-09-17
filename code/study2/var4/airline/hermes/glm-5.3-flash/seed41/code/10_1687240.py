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
DATE_COLS = ["Month", "DayofMonth", "DayOfWeek"]
# interaction categoricals: entity x scheduled-departure-hour
INTER_PAIRS = [
    ("UniqueCarrier", "hour"),
    ("Origin", "hour"),
    ("Dest", "hour"),
    ("DayOfWeek", "hour"),
]

# ensemble members: (drop list, overrides) — two feature views x regularization levels.
# averaging decorrelated members (different feature views / regularization / depth) lifts
# the blend well above any single member. 3 full-view + 3 airport-view members.
_MEM = dict(n_estimators=1200, learning_rate=0.03, max_depth=8, min_child_weight=20, colsample_bytree=0.7)
_AP_DROP = ("Month", "DayofMonth", "DayOfWeek")
_DOM_DROP = ("Month", "DayofMonth")
_SLOW = {**_MEM, "learning_rate": 0.02, "n_estimators": 1800}
MEMBERS = [
    ((),       {**_MEM, "reg_alpha": 20.0}),
    (_AP_DROP, {**_MEM, "reg_alpha": 0.3}),
    (_AP_DROP, {**_MEM, "reg_alpha": 1.0, "max_depth": 10, "min_child_weight": 30}),
    (_DOM_DROP, {**_MEM, "reg_alpha": 3.0}),
    ((),       {**_SLOW, "reg_alpha": 20.0}),
]


# --- features -----------------------------------------------------------------
def _fe(df: pd.DataFrame, drop=()) -> pd.DataFrame:
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
    for d in drop:
        X = X.drop(columns=[d])
    return X


CAT_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
_TRAIN_FULL = _fe(train)
INTER_LEVELS = {
    (_a, _b): pd.Index(
        sorted((train[_a].astype(str) + "_" + _TRAIN_FULL[_b].astype(str)).unique())
    )
    for _a, _b in INTER_PAIRS
}


def prepare(df: pd.DataFrame, drop=()) -> pd.DataFrame:
    """ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows."""
    X = _fe(df, drop)
    for c in [c for c in CAT_COLS if c in X.columns]:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    for a, b in INTER_PAIRS:
        if a not in X.columns or b not in X.columns:
            continue
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

t0 = time.time()
y_train = to_y(train)
models = []
for drop, spec in MEMBERS:
    m = xgb.XGBClassifier(**_BASE_PARAMS, **spec)
    m.fit(prepare(train, drop), y_train)
    models.append((m, drop))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    p = np.mean([m.predict_proba(prepare(df, drop))[:, 1] for m, drop in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
