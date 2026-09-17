"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
Cyc = dict(Month=("c-", 12), DayofMonth=("c-", 31), DayOfWeek=("c-", 7))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c, (prefix, _max) in Cyc.items():
        X[c] = _num(df[c]).astype("float32")
    dt = df["DepTime"].astype("float32")
    X["DepTime"] = dt
    X["Distance"] = df["Distance"].astype("float32")
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=18,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=1,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, model.best_iteration + 1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
