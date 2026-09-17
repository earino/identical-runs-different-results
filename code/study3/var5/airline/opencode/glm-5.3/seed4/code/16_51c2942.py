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
cat_levels = {}


def _num(s: pd.Series) -> pd.Series:
    """c-<n> string -> int."""
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 24)
    X["Month"] = _num(df["Month"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DepTime"] = dep
    X["Hour"] = hour
    X["Mins"] = hour * 60 + dep % 100
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# fit categorical levels on TRAIN ONLY
for c in CAT_COLS:
    cat_levels[c] = pd.Index(sorted(train[c].dropna().unique()))

# --- model: bagged ensemble of XGBoost models ---------------------------------
N_MODELS = 20
BAG_FRAC = 0.8
models = []
t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
rng = np.random.RandomState(SEED)
for k in range(N_MODELS):
    idx = rng.choice(len(train), size=int(BAG_FRAC * len(train)), replace=False)
    m = xgb.XGBClassifier(
        n_estimators=60,
        max_depth=20,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        colsample_bytree=0.5,
        random_state=SEED + k,
        n_jobs=N_JOBS,
    )
    m.fit(X_all.iloc[idx], y_all[idx])
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
