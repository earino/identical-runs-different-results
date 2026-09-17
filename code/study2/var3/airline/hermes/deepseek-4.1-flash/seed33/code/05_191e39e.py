"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
INTER_COLS = ["CarrierHour"]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> feature frame. Month/DayofMonth are deliberately excluded: calendar-position
    effects do not carry from 2005 to 2006 and including them measurably hurt."""
    X = pd.DataFrame(index=df.index)

    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).astype(float)
    hour = hour.where(hour < 24, 0.0)              # 2400 == midnight, >2400 == junk
    minute = (dep % 100).astype(float)
    sod = hour.fillna(-1.0) * 60 + minute.fillna(-1.0)
    X["dep_hour"] = hour.fillna(-1.0)
    X["dep_minute"] = minute.fillna(-1.0)
    X["dep_sod"] = sod
    X["dep_sin"] = np.sin(2 * np.pi * sod / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * sod / 1440.0)

    X["DayOfWeek"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_distance"] = np.log1p(dist)

    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    # carrier-specific departure-hour banks: a low-cardinality interaction that transfers across years
    X["CarrierHour"] = X["UniqueCarrier"] + "_" + X["dep_hour"].astype(int).astype(str)
    return X


_SAMPLE = _features(train)
_CAT_LEVELS = {c: pd.Index(sorted(_SAMPLE[c].dropna().unique())) for c in CAT_COLS + INTER_COLS}
FEATURE_COLS = list(_SAMPLE.columns)
del _SAMPLE


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _features(df)[FEATURE_COLS].copy()
    for c in CAT_COLS + INTER_COLS:
        X[c] = pd.Categorical(X[c], categories=_CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models -------------------------------------------------------------------
X_train = prepare(train)
y_train = to_y(train)

t0 = time.time()
models = []
for depth in (3, 6, 9, 12, 15, 18):
    m = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=depth,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
