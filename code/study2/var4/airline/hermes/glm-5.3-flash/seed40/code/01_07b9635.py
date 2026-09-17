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

# category levels fit on TRAIN only; unseen levels in eval/holdout -> code -1 -> missing
CAT_COLS = ("UniqueCarrier", "Origin", "Dest")
CAT_LEVELS = {c: sorted(train[c].dropna().astype(str).unique().tolist()) for c in CAT_COLS}


# --- features -----------------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() applies it to unseen rows too."""
    X = pd.DataFrame(index=df.index)

    # Month / DayofMonth / DayOfWeek come as "c-<n>" strings -> numeric
    for c in ("Month", "DayofMonth", "DayOfWeek"):
        X[c] = pd.to_numeric(df[c].astype(str).str.lstrip("c"), errors="coerce")

    # DepTime: scheduled departure as hhmm integer (values like 1..959 lack a leading zero).
    s = df["DepTime"].fillna(0).astype("int64").astype(str).str.zfill(4)
    hh = pd.to_numeric(s.str[:2], errors="coerce")
    mm = pd.to_numeric(s.str[2:4], errors="coerce")
    hh = hh.where((hh >= 0) & (hh < 24), np.nan)
    mm = mm.where((mm >= 0) & (mm < 60), np.nan)
    mins = hh * 60 + mm
    X["dep_hour"] = hh
    X["dep_tod"] = np.floor(mins / 15)  # 0..95 quarter of day
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["red_eye"] = np.where(hh.isna(), -1, ((hh >= 20) | (hh <= 5)).astype(float))

    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 9,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": SEED,
    "nthread": N_JOBS,
}


def fit_model(X, y, params=None, num_boost_round=600):
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    p = dict(PARAMS)
    if params:
        p.update(params)
    return xgb.train(p, dtrain, num_boost_round, verbose_eval=False)


t0 = time.time()
X_train = prepare(train)
y_train = to_y(train)
model = fit_model(X_train, y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dp = xgb.DMatrix(prepare(df), enable_categorical=True)
    return model.predict(dp)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
