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


def _num(col: pd.Series) -> pd.Series:
    return col.str.split("-").str[1].astype(int)


# --- features -----------------------------------------------------------------
RAW_NUM = ["DepTime", "Distance"]
CYC = {"Month": 12, "DayofMonth": 31, "DayOfWeek": 7}
CAT = ["UniqueCarrier", "Origin", "Dest"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in RAW_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["Dist_log"] = np.log1p(X["Distance"])
    for c, period in CYC.items():
        n = _num(df[c])
        X[c + "_n"] = n
        X[c + "_sin"] = np.sin(2 * np.pi * n / period)
        X[c + "_cos"] = np.cos(2 * np.pi * n / period)
    dt = X["DepTime"]
    hour = dt // 100
    minute = dt % 100
    tod = hour * 60 + minute
    X["Hour"] = hour
    X["Tod_sin"] = np.sin(2 * np.pi * tod / 1440)
    X["Tod_cos"] = np.cos(2 * np.pi * tod / 1440)
    for c in CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of XGB models --------------------------------------
N_MODELS = 12
SUB = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85] * 2
models = [
    xgb.XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=SUB[s % len(SUB)],
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    for s in range(N_MODELS)
]

t0 = time.time()
X_tr = prepare(train)
y_tr = to_y(train)
for m in models:
    m.fit(X_tr, y_tr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
