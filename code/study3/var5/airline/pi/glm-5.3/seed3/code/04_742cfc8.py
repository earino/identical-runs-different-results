"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


# carrier-hour interaction levels fit on TRAINING data only
_hr_tr = np.floor(train["DepTime"].astype(float) / 100) % 24
car_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "|" + _hr_tr.astype(int).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    dep = df["DepTime"].astype(float)
    hour = np.floor(dep / 100) % 24
    X["hour"] = hour
    X["mins"] = hour * 60 + (dep - 100 * np.floor(dep / 100))
    # time-of-day harmonics: smooth, year-stable representation of the daily delay pattern
    t = 2 * np.pi * X["mins"] / 1440
    for k in range(1, 9):
        X[f"sin{k}"] = np.sin(k * t)
        X[f"cos{k}"] = np.cos(k * t)
    # carrier x hour-of-day interaction: evening hub congestion differs by carrier
    X["car_hour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "|" + hour.astype(int).astype(str),
                                   categories=car_hour_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed-averaged ensemble of XGBoost models ---------------------------
X_train, y_train = prepare(train), to_y(train)
models = []
t0 = time.time()
for seed in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        colsample_bynode=0.2,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
