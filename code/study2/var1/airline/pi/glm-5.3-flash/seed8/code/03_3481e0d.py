"""XGBoost binary classifier for airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
RAW_CAT = ["UniqueCarrier", "Origin", "Dest"]


def _num_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric engineered features from raw columns (no fitted state)."""
    month = df["Month"].str[2:].astype(int)
    day = df["DayofMonth"].str[2:].astype(int)
    dow = df["DayOfWeek"].str[2:].astype(int)
    dep_min = (df["DepTime"] // 100 % 24) * 60 + df["DepTime"] % 100  # wraps 2400+/odd codes
    hour = dep_min // 60
    out = pd.DataFrame(
        {
            "month": month,
            "day": day,
            "dow": dow,
            "dep_min": dep_min,
            "hour": hour,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
            "dow_sin": np.sin(2 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2 * np.pi * dow / 7.0),
            "distance": df["Distance"].astype(float),
            "log_dist": np.log1p(df["Distance"].astype(float)),
            "is_weekend": (dow >= 6).astype(int),
        },
        index=df.index,
    )
    return out


feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}
cat_levels["route"] = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _num_parts(df)
    X["route"] = pd.Categorical(
        df["Origin"] + "_" + df["Dest"], categories=cat_levels["route"]
    )  # unseen levels -> NaN
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
)

# --- model --------------------------------------------------------------------
t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=200,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    **PARAMS,
)
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
