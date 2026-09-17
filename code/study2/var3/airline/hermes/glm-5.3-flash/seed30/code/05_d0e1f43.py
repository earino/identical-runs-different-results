"""XGBoost binary classifier for airline delay. Exp 12: day-of-year seasonality + minutes-of-day numerics."""
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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
ENG_COLS = ["DepHour", "DepTimeSin", "DepTimeCos", "DistanceLog"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    t = pd.to_numeric(df["DepTime"], errors="coerce").astype("float64")
    t_mod = t % 2400  # 2400-2435 midnight-crossings -> 0-35
    hh = (t_mod // 100).clip(0, 23)
    mm = (t_mod % 100).clip(0, 59)
    X["DepHour"] = hh
    X["DepMin"] = hh * 60.0 + mm
    X["DepTimeSin"] = np.sin(2 * np.pi * X["DepMin"] / 1440.0)
    X["DepTimeCos"] = np.cos(2 * np.pi * X["DepMin"] / 1440.0)
    X["DistanceLog"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce").astype("float64"))
    month = pd.to_numeric(df["Month"].str.slice(2), errors="coerce")
    day = pd.to_numeric(df["DayofMonth"].str.slice(2), errors="coerce")
    X["SeasonSin"] = np.sin(2 * np.pi * (month * 30.4 + day) / 365.0)
    X["SeasonCos"] = np.cos(2 * np.pi * (month * 30.4 + day) / 365.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=8000,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=200,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(
    prepare(train),
    to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=False,
)
print(f"Training time: {time.time() - t0:.1f}s best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
