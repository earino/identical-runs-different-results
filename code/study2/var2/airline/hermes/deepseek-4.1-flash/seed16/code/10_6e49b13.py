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

# --- feature specification -----------------------------------------------------
# Raw columns: Month/DayofMonth/DayOfWeek are "c-<n>" strings -> ordinal ints.
# DepTime is hhmm (a few dirty values >2400) -> split into hour/minute + cyclic time-of-day.
NUM_COLS = ["Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]

CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
# max values seen in training, used to clip dirty minutes/hours in unseen data
DEP_MAX = int(train["DepTime"].clip(lower=0).max())


def _hour_series(s: pd.Series) -> pd.Series:
    dt = pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").clip(lower=0, upper=DEP_MAX)
    return (dt // 100) % 24


CARRIER_HOUR_LEVELS = pd.Index(
    sorted((train["UniqueCarrier"].astype(str) + "_" + _hour_series(train["DepTime"]).astype(str)).unique())
)


def _ordinal(s: pd.Series) -> pd.Series:
    """'c-12' -> 12 (int). Non-matching values -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = _ordinal(df["Month"]).astype("float32")
    X["month_cat"] = pd.Categorical(_ordinal(df["Month"]).fillna(-1).astype(int), categories=list(range(1, 13)))
    X["dom"] = _ordinal(df["DayofMonth"]).astype("float32")
    dow = _ordinal(df["DayOfWeek"])
    X["dow"] = dow.astype("float32")
    X["dow_cat"] = pd.Categorical(dow.fillna(-1).astype(int), categories=list(range(1, 8)))
    X["is_weekend"] = (dow >= 6).astype("int8")

    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype("int64").clip(lower=0, upper=DEP_MAX)
    hour = (dt // 100) % 24
    minute = dt % 100
    tod = (hour * 60 + minute).astype("int32")
    X["dep_hour"] = hour.astype("int8")
    X["dep_hour_cat"] = pd.Categorical(hour.astype(int), categories=list(range(24)))
    X["dep_minute"] = minute.astype("int8")
    X["dep_tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0).astype("float32")
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0).astype("float32")

    for c in NUM_COLS:
        X[c.lower()] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    X["log_distance"] = np.log1p(X["distance"].clip(lower=0))

    for c in CAT_COLS:
        X[c.lower()] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    # carrier x hour-of-day interaction as a single categorical
    ch = df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
    X["carrier_hour_cat"] = pd.Categorical(ch, categories=CARRIER_HOUR_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small seed/jitter ensemble (all XGBoost) ---------------------------
PARAMS = dict(
    n_estimators=400,
    max_depth=0,
    learning_rate=0.05,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    max_cat_threshold=128,
    grow_policy="lossguide",
    max_leaves=128,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
ENSEMBLE = [
    {"seed": SEED + 0, "colsample_bytree": 0.8},
    {"seed": SEED + 1, "colsample_bytree": 0.7, "subsample": 0.7},
    {"seed": SEED + 2, "colsample_bytree": 0.9},
    {"seed": SEED + 3, "max_depth": 7, "min_child_weight": 40, "colsample_bytree": 0.8},
    {"seed": SEED + 4, "max_depth": 5, "colsample_bytree": 0.8},
    {"seed": SEED + 5, "max_depth": 6, "min_child_weight": 10, "colsample_bytree": 0.6, "subsample": 0.7},
    {"seed": SEED + 6, "max_depth": 8, "min_child_weight": 60, "colsample_bytree": 0.7, "reg_lambda": 5.0},
    {"seed": SEED + 7, "n_estimators": 700, "learning_rate": 0.03, "colsample_bytree": 0.8},
]

X_train = prepare(train)
y_train = to_y(train)

t0 = time.time()
models = []
for over in ENSEMBLE:
    over = dict(over)
    seed = over.pop("random_state", None) or over.pop("seed", None)
    params = dict(PARAMS, **over)
    params["random_state"] = seed if seed is not None else PARAMS.get("random_state", SEED)
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
