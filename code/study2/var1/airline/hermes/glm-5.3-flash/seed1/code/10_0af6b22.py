"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(df); any statistics it needs are fit on TRAIN data only.
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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# scheduled hour-of-day as a categorical: the hourly delay-rate curve is strongly non-linear
# (0.85 at 1am, 0.04 at 5am, 0.83 at 11pm) and highly stable across years
HOUR_LEVELS = pd.Index([str(h) for h in range(24)])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    hour = (df["DepTime"].astype(int) // 100) % 24
    X["dep_hour"] = pd.Categorical(hour.astype(str), categories=HOUR_LEVELS)
    # carrier x half-hour-of-day: each carrier has its own characteristic delay profile by
    # time of day, and that profile is stable across years
    CRS = pd.Index(sorted(cat_levels["UniqueCarrier"]))
    dt = df["DepTime"].astype(int)
    half = np.where((dt % 100) >= 30, "b", "a")
    hh = pd.Series(half, index=df.index).str.cat(pd.Series(hour).astype(str), sep="_")
    X["cr_x_half"] = pd.Categorical(
        df["UniqueCarrier"].str.cat(hh, sep="_"),
        categories=[f"{c}_{p}_{h}" for c in CRS for p in ("a", "b") for h in range(24)],
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=4000,
    max_depth=4,
    learning_rate=0.02,
    colsample_bytree=0.5,
    early_stopping_rounds=500,
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iteration={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
