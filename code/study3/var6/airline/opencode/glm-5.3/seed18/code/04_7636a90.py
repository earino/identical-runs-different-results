"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_RAW}


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    h = (t // 100).fillna(0)
    m = (t % 100).fillna(0)
    frac = h + m / 60.0
    X["hour"] = h
    X["minute"] = m
    X["frac_hour"] = frac
    X["sin_h"] = np.sin(2 * np.pi * frac / 24.0)
    X["cos_h"] = np.cos(2 * np.pi * frac / 24.0)
    X["sin_m"] = np.sin(2 * np.pi * month / 12.0)
    X["cos_m"] = np.cos(2 * np.pi * month / 12.0)
    X["sin_d"] = np.sin(2 * np.pi * day / 31.0)
    X["cos_d"] = np.cos(2 * np.pi * day / 31.0)
    X["sin_w"] = np.sin(2 * np.pi * dow / 7.0)
    X["cos_w"] = np.cos(2 * np.pi * dow / 7.0)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_dist"] = np.log1p(X["distance"])
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=6000,
    max_depth=14,
    learning_rate=0.02,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_lambda=1.0,
    reg_alpha=3.0,
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)
model.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
