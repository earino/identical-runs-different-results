"""XGBoost binary classifier on airline delays. Only file the agent edits.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering. Baseline column order preserved; extras appended."""
    X = pd.DataFrame(index=df.index)
    month = df["Month"].str[2:].astype(np.int32)
    dom = df["DayofMonth"].str[2:].astype(np.int32)
    X["doy"] = month * 31 + dom
    for c in CAT_COLS:
        X[c] = df[c]
    X["DepTime"] = df["DepTime"].to_numpy()
    X["Distance"] = df["Distance"].to_numpy()
    return X


# fit category levels on training data only
_fit = _engineer(train)
cat_levels = {c: pd.Index(sorted(_fit[c].dropna().unique())) for c in CAT_COLS}
FEATURE_NAMES = list(_fit.columns)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = _engineer(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[FEATURE_NAMES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=60,
    learning_rate=0.12,
    max_depth=5,
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
