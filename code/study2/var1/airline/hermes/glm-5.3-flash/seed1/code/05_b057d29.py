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


# --- train-only statistics, closed over by prepare() --------------------------------------
GLOBAL_RATE = float((train[TARGET] == POSITIVE).mean())
K = 60.0  # smoothing strength in pseudo-observations

ORIGIN_N = train.groupby("Origin").size()
ORIGIN_R = train.groupby("Origin").apply(lambda g: (g[TARGET] == POSITIVE).mean())
DEST_N = train.groupby("Dest").size()
DEST_R = train.groupby("Dest").apply(lambda g: (g[TARGET] == POSITIVE).mean())
ROUTE_N = train.groupby(["Origin", "Dest"]).size()
ROUTE_R = train.groupby(["Origin", "Dest"]).apply(lambda g: (g[TARGET] == POSITIVE).mean())


def _smooth_rate(stat_r, stat_n, key):
    n = key.map(stat_n).fillna(0).to_numpy(dtype=float)
    r = key.map(stat_r).fillna(GLOBAL_RATE).to_numpy(dtype=float)
    return (r * n + GLOBAL_RATE * K) / (n + K)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # airport / route delay priors from train only, smoothed towards the global rate
    X["origin_rate"] = _smooth_rate(ORIGIN_R, ORIGIN_N, df["Origin"])
    X["dest_rate"] = _smooth_rate(DEST_R, DEST_N, df["Dest"])
    key = pd.MultiIndex.from_arrays([df["Origin"], df["Dest"]])
    n = pd.Series(ROUTE_N).reindex(key).fillna(0).to_numpy(dtype=float)
    r = pd.Series(ROUTE_R).reindex(key).fillna(GLOBAL_RATE).to_numpy(dtype=float)
    X["route_rate"] = (r * n + GLOBAL_RATE * K) / (n + K)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.1,
    early_stopping_rounds=50,
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
