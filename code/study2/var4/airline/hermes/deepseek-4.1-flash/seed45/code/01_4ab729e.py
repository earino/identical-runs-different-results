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

# --- features -----------------------------------------------------------------
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
OBJ_COLS = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in OBJ_COLS if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
NUM_COLS = [c for c in RAW_COLS if c not in OBJ_COLS]

# c-<n> encoded ordinal columns (Month, DayofMonth, DayOfWeek)
C_NUM = {c: c for c in cat_cols if train[c].str.startswith("c-", na=False).all()}


def _c_to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        if c in C_NUM:
            v = _c_to_int(df[c]).astype("float64")
            if c == "Month":
                X["month"] = v
                X["month_sin"] = np.sin(2 * np.pi * v / 12.0)
                X["month_cos"] = np.cos(2 * np.pi * v / 12.0)
            elif c == "DayOfWeek":
                X["dow"] = v
                X["dow_sin"] = np.sin(2 * np.pi * v / 7.0)
                X["dow_cos"] = np.cos(2 * np.pi * v / 7.0)
            elif c == "DayofMonth":
                X["dom"] = v
                X["dom_sin"] = np.sin(2 * np.pi * v / 31.0)
                X["dom_cos"] = np.cos(2 * np.pi * v / 31.0)
        else:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    # scheduled departure: hhmm -> minutes since midnight (+ cyclical position in the day)
    hh = np.floor(X["DepTime"] / 100.0)
    mm = X["DepTime"] - hh * 100.0
    dep = np.where((hh >= 0) & (hh <= 23) & (mm >= 0) & (mm <= 59), hh * 60.0 + mm, np.nan)
    X["dep_min"] = dep
    X["dep_hour"] = dep / 60.0
    X["dep_hour_sin"] = np.sin(2 * np.pi * dep / 1440.0)
    X["dep_hour_cos"] = np.cos(2 * np.pi * dep / 1440.0)
    X = X.drop(columns=["DepTime"])
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
