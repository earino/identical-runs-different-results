"""XGBoost binary classifier for airline departure delay prediction.

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
obj_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]
            and (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in obj_cols}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 24)
    minute = dep % 100
    mins = (hour * 60 + minute).clip(0, 1440)
    X["DepTime"] = dep
    X["mins"] = mins
    X["sin_day"] = np.sin(2 * np.pi * mins / 1440)
    X["cos_day"] = np.cos(2 * np.pi * mins / 1440)
    X["sin_week"] = np.sin(2 * np.pi * (df["DayOfWeek"].str.extract(r"(\d+)").astype(float)[0] - 1) / 7)
    X["cos_week"] = np.cos(2 * np.pi * (df["DayOfWeek"].str.extract(r"(\d+)").astype(float)[0] - 1) / 7)
    X["sin_year"] = np.sin(2 * np.pi * df["DayofMonth"].str.extract(r"(\d+)").astype(float)[0] / 31)
    X["cos_year"] = np.cos(2 * np.pi * df["DayofMonth"].str.extract(r"(\d+)").astype(float)[0] / 31)
    X["hour"] = hour.astype(int)
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels)
    for c in obj_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model (baseline config: small shallow ensemble) ---------------------------
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
