"""XGBoost binary classifier for airline delay. Agent-edited file.

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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- encoders fitted on TRAIN only --------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in obj_cols}
# calendar: month/day strings -> day-of-year lookup fitted on train
MONTHS = {f"c-{i}": i for i in range(1, 13)}
DOM = {f"c-{i}": i for i in range(1, 32)}
CUMDAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])  # non-leap 2005


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() reproduces it on unseen data."""
    X = df[feature_cols].copy()
    for c in obj_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # time of day: DepTime is hhmm, may exceed 2400 (next-day roll)
    v = X["DepTime"].astype("float64") % 2400
    hour = (v // 100).astype("int32")
    minute = (v % 100).astype("int32")
    mins = hour * 60 + minute
    X["dep_hour"] = pd.Categorical(hour.astype(str), categories=[str(i) for i in range(24)])
    X["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["dep_minute"] = minute
    # calendar numerics + seasonality
    m = X["Month"].astype(str).map(MONTHS)
    d = X["DayofMonth"].astype(str).map(DOM)
    dow = X["DayOfWeek"].astype(str).map(lambda s: int(str(s).split("-")[1]) if pd.notna(s) else np.nan)
    X["doy"] = CUMDAYS[(m - 1).fillna(0).astype("int32").to_numpy()] + d
    X["doy_sin"] = np.sin(2 * np.pi * X["doy"] / 365)
    X["doy_cos"] = np.cos(2 * np.pi * X["doy"] / 365)
    X["month_num"] = m.astype("float64")
    X["dow_num"] = dow.astype("float64")
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["log_dist"] = np.log1p(X["Distance"].astype("float64"))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=100,
)

t0 = time.time()
X_tr, X_val, y_tr, y_val = train_test_split(
    prepare(train), to_y(train), test_size=0.1, random_state=SEED, stratify=to_y(train)
)
model = xgb.XGBClassifier(**PARAMS)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
best_iters = int(model.best_iteration)
print(f"best_iteration: {best_iters}")
# retrain on the full training set with the round count found by early stopping
model = xgb.XGBClassifier(**{**PARAMS, "n_estimators": best_iters + 1, "early_stopping_rounds": None})
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
