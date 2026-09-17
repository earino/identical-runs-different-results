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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _parse_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _parse_c(df["Month"])
    X["DayofMonth"] = _parse_c(df["DayofMonth"])
    X["DayOfWeek"] = _parse_c(df["DayOfWeek"])
    dep = df["DepTime"].astype(float)
    X["DepTime"] = dep
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["mins_day"] = dep // 100 * 60 + dep % 100
    X["Distance"] = df["Distance"].astype(float)
    X["log_Distance"] = np.log1p(df["Distance"].astype(float))
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = df[c]
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr_all, ytr_all = prepare(train), to_y(train)

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xtr_all, ytr_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
