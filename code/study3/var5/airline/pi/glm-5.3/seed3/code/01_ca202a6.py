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
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]


def _int_from_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"^\D+", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _int_from_c(df["Month"])
    day = _int_from_c(df["DayofMonth"])
    dow = _int_from_c(df["DayOfWeek"])
    dep = df["DepTime"].astype(float)
    hour = np.floor(dep / 100) % 24
    minute = dep - 100 * np.floor(dep / 100)
    mins = hour * 60 + minute
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    X["hour"] = hour
    X["mins"] = mins
    X["sin_d"] = np.sin(2 * np.pi * mins / 1440)
    X["cos_d"] = np.cos(2 * np.pi * mins / 1440)
    X["sin_m"] = np.sin(2 * np.pi * month / 12)
    X["cos_m"] = np.cos(2 * np.pi * month / 12)
    X["distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    X["dep_raw"] = dep
    X["Route"] = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
    X["Month"] = month
    X["DayofMonth"] = day
    X["DayOfWeek"] = dow
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# categorical levels fit on TRAINING data only
_tmp = pd.DataFrame(index=train.index)
_tmp["Route"] = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
cat_levels = {}
for c in CAT_COLS:
    if c == "Route":
        cat_levels[c] = pd.Index(sorted(_tmp["Route"].unique()))
    else:
        cat_levels[c] = pd.Index(sorted(train[c].dropna().unique()))

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
