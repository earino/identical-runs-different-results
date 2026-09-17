"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives inside prepare(), which predict_proba() calls on unseen rows.
"""
import json
import os
import re
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

# --- feature configuration (derived from the TRAINING frame only) --------------
RAW_FEATURES = [c for c in train.columns if c not in ID_COLS + [TARGET]]

_C_TOKEN = re.compile(r"^c-(\d+)$")


def _is_c_token(series: pd.Series) -> bool:
    """True if every non-null value looks like 'c-<int>' (i.e. an ordinal coded as a string)."""
    vals = series.dropna().unique()
    return len(vals) > 0 and all(_C_TOKEN.match(str(v)) for v in vals)


# string columns that are really ordinal ints: Month / DayofMonth / DayOfWeek
ORD_COLS = [c for c in RAW_FEATURES if pd.api.types.is_string_dtype(train[c])
            or pd.api.types.is_object_dtype(train[c])]
ORD_COLS = [c for c in ORD_COLS if _is_c_token(train[c])]

# remaining string columns are true categoricals
CAT_COLS = [c for c in RAW_FEATURES
            if (pd.api.types.is_string_dtype(train[c]) or pd.api.types.is_object_dtype(train[c]))
            and c not in ORD_COLS and train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# DepTime (hhmm) -> hour / minute / minutes-of-day. Some encodings use 2400 for midnight.
DEPTIME = "DepTime" if "DepTime" in RAW_FEATURES else None


def _hhmm(df: pd.DataFrame) -> tuple:
    t = pd.to_numeric(df[DEPTIME], errors="coerce")
    t = t.where(t < 2400, t - 2400)
    hour = np.floor(t / 100.0)
    minute = t - hour * 100.0
    hour = hour.where((hour >= 0) & (hour <= 23), np.nan)
    minute = minute.where((minute >= 0) & (minute <= 59), np.nan)
    return hour, minute


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[RAW_FEATURES].copy()
    for c in ORD_COLS:
        X[c] = pd.to_numeric(X[c].astype(str).str.replace("^c-", "", regex=True), errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    if DEPTIME is not None:
        hour, minute = _hhmm(df)
        X["dep_hour"] = hour
        X["dep_minute"] = minute
        X["dep_minofday"] = hour * 60.0 + minute
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
