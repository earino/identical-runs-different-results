"""XGBoost binary classifier for the airline delay task.

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
DEPTH = int(os.environ.get("EXP_DEPTH", "3"))
N_TREES = int(os.environ.get("EXP_TREES", "500"))
LR = float(os.environ.get("EXP_LR", "0.05"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature layout -----------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _dom(s: pd.Series) -> pd.Series:
    # "c-<n>" -> <n>, day of month 1..31
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["Distance"] = _num(df["Distance"])
    v = _num(df["DepTime"])
    hr = (v // 100).clip(0, 23)
    mn = (v % 100).clip(0, 59)
    m = hr * 60 + mn  # minutes since midnight, 2400+ wraps to 0
    X["DepHour"] = hr
    X["DepMin"] = m
    ang = 2 * np.pi * m / 1440.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    X["DepHourSin"] = np.sin(2 * np.pi * hr / 24.0)
    X["DepHourCos"] = np.cos(2 * np.pi * hr / 24.0)
    dom = _dom(df["DayofMonth"])
    X["DomSin"] = np.sin(2 * np.pi * dom / 31.0)
    X["DomCos"] = np.cos(2 * np.pi * dom / 31.0)
    X["DepMinInHr"] = mn
    X["OnSlot"] = ((mn % 15) == 0).astype(float)
    X["LogDistance"] = np.log1p(X["Distance"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=N_TREES,
    max_depth=DEPTH,
    learning_rate=LR,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Config: depth={DEPTH} trees={N_TREES} lr={LR}")
print(f"Training time: {time.time() - t0:.1f}s")

Xint = prepare(train)
p_int = model.predict_proba(Xint)[:, 1]
print(f"Internal 2005 AUC (resub): {roc_auc_score(to_y(train), p_int):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
