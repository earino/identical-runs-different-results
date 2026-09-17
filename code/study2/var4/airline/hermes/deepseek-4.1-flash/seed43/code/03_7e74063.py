"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(), which predict_proba() calls on unseen rows.
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

RAW_CATS = ["UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["Distance"]

feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]


def _code(s: pd.Series) -> pd.Series:
    """'c-4' -> 4 (the raw values are integer codes rendered as strings)."""
    return pd.to_numeric(s.astype(str).str.split("-").str[-1], errors="coerce")


# levels / lookups fitted on the training split only
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
route_levels = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).dropna().unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _code(df["Month"])
    dom = _code(df["DayofMonth"])
    dow = _code(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100) % 24
    minute = dep % 100
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = hour + minute / 60.0
    X["is_weekend"] = (dow >= 6).astype(float)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["Distance"])
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
ytr = to_y(train)
Xtr_all = prepare(train)

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
model.fit(Xtr_all, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
