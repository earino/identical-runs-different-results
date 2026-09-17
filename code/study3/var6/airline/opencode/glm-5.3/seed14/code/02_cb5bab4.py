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
CAT = ["UniqueCarrier", "Origin", "Dest", "Route"]

train["Route"] = train["Origin"] + "_" + train["Dest"]
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
        X[c] = pd.Categorical(df[c] if c != "Route" else df["Origin"] + "_" + df["Dest"], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.07,
    subsample=0.8,
    colsample_bytree=0.8,
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
