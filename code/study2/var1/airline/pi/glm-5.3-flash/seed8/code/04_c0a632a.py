"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: one-hot encoding of categoricals (robust across the 2005->2006 year shift, unlike native
categorical partitions), numeric time features, carrier-x-hour interaction, and a 5-fold bagged
ensemble of XGBoost models.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- fitted state (train only) --------------------------------------------------
CATS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CATS}
CH_LEVELS = pd.Index(sorted((train["UniqueCarrier"] + "_" + (train["DepTime"] // 100 % 24).astype(str)).unique()))


# --- features (all inside prepare, used identically by training and predict_proba) ---
def numeric_parts(df: pd.DataFrame) -> pd.DataFrame:
    month = df["Month"].str[2:].astype(int)
    dow = df["DayOfWeek"].str[2:].astype(int)
    day = df["DayofMonth"].str[2:].astype(int)
    dep_min = (df["DepTime"] // 100 % 24) * 60 + df["DepTime"] % 100  # wraps 2400+/odd codes
    hour = dep_min // 60
    return pd.DataFrame(
        {
            "month": month,
            "day": day,
            "dow": dow,
            "dep_min": dep_min,
            "hour": hour,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "log_dist": np.log1p(df["Distance"].astype(float)),
            "is_weekend": (dow >= 6).astype(int),
        },
        index=df.index,
    )


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = numeric_parts(df)
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"].astype(float)
    for c in CATS:  # unseen levels -> all-zero dummy row
        X = pd.concat(
            [X, pd.get_dummies(df[c].astype(pd.CategoricalDtype(categories=LEVELS[c])), prefix=c).astype(float)],
            axis=1,
        )
    ch = df["UniqueCarrier"] + "_" + (df["DepTime"] // 100 % 24).astype(str)
    X = pd.concat(
        [X, pd.get_dummies(ch.astype(pd.CategoricalDtype(categories=CH_LEVELS)), prefix="ch").astype(float)], axis=1
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 5-fold bagged ensemble -----------------------------------------------
PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=SEED,
    n_jobs=N_JOBS,
)
N_FOLDS = 5

t0 = time.time()
X = prepare(train)
y = to_y(train)
models = []
for a, b in KFold(N_FOLDS, shuffle=True, random_state=1).split(X):
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(X.iloc[a], y[a])
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} x {PARAMS['n_estimators']} trees, depth {PARAMS['max_depth']})")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
