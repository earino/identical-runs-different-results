"""XGBoost binary classifier for airline delay. Only file the agent edits.

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

# --- stats fitted on training data only ---------------------------------------
CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TR_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CATS}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace("c-", "", regex=False), errors="coerce").astype(float)


def _hour_str(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    return ((dt // 100) % 24).astype(int).astype(str)


_org_hour_levels = (train["Origin"].astype(str) + "_" + _hour_str(train)).unique()


# --- feature engineering (everything inside prepare) ---------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=ID_COLS + [TARGET], errors="ignore")
    X = pd.DataFrame(index=df.index)
    # calendar as numeric (beats categorical on time-separated eval)
    X["Month"] = _num(df["Month"])
    X["DayOfWeek"] = _num(df["DayOfWeek"])
    X["DayofMonth"] = _num(df["DayofMonth"])
    # IDs as categoricals with train-fitted levels (unseen -> NaN)
    for c in ("UniqueCarrier", "Origin", "Dest"):
        X[c] = pd.Categorical(df[c].astype(str), categories=TR_LEVELS[c])
    # time-of-day features
    dt = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    X["DepTime"] = dt
    X["DepHour"] = (dt // 100) % 24
    X["DepAbs"] = (dt // 100) * 60 + dt % 100  # wraps past-midnight departures to late hours
    X["LateNight"] = ((dt >= 1900) | (dt < 600)).astype(float)
    ang = 2 * np.pi * X["DepAbs"] / 1440.0
    X["DepSin"], X["DepCos"] = np.sin(ang), np.cos(ang)
    # origin x scheduled-hour interaction (strongest single feature found)
    hs = _hour_str(df)
    X["OrgHour"] = pd.Categorical(df["Origin"].astype(str) + "_" + hs, categories=_org_hour_levels)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=60,
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
