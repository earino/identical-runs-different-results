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

# --- schema (from column names only; no data statistics) -----------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
RAW_CAL = ["Month", "DayofMonth", "DayOfWeek"]  # replaced by numeric/cyclic versions in prepare()
BASE = [c for c in train.columns if c not in ID_COLS + [TARGET] + CAT_COLS + RAW_CAL]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num(series: pd.Series) -> pd.Series:
    """'c-7' -> 7 (calendar fields are stored as c-<n> strings)."""
    return series.astype(str).str.replace("c-", "", regex=False).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[BASE + CAT_COLS].copy()

    dep = df["DepTime"].to_numpy(dtype=float)
    hour = np.floor(dep / 100.0)              # after-midnight flights are stored as 24xx..26xx
    tod = hour * 60.0 + (dep - hour * 100.0)  # minutes since midnight
    tod = np.where(tod >= 24 * 60.0, tod - 24 * 60.0, tod)
    X["dep_hour"] = np.floor(tod / 60.0)
    X["dep_minute"] = tod % 60.0
    X["dep_tod"] = tod
    X["dep_sin"] = np.sin(2.0 * np.pi * tod / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * tod / 1440.0)

    X["month_n"] = _num(df["Month"])
    X["dom_n"] = _num(df["DayofMonth"])
    X["dow_n"] = _num(df["DayOfWeek"])
    X["is_weekend"] = (X["dow_n"] >= 6).astype(int)
    X["dist_log"] = np.log1p(df["Distance"])

    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


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
