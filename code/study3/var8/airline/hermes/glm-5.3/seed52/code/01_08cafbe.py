"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
raw_feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [
    c
    for c in raw_feature_cols
    if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])
]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in raw_feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# Smoothed target encoding of categorical columns (fit on train only).
TE_SMOOTH = 20.0
te_maps: dict[str, dict] = {}


def _fit_target_encodings(df: pd.DataFrame, y: np.ndarray) -> None:
    global_mean = float(y.mean())
    for c in cat_cols:
        stats = pd.DataFrame({"cat": df[c], "y": y}).groupby("cat")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + TE_SMOOTH * global_mean) / (stats["count"] + TE_SMOOTH)
        te_maps[c] = enc.to_dict()
        te_maps[c]["__global__"] = global_mean


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dep = df["DepTime"].astype(float)
    hour = np.clip(np.floor(dep / 100) % 24, 0, 23)
    minute = dep - 100 * np.floor(dep / 100)
    out = pd.DataFrame(index=df.index)
    out["hour"] = hour
    out["minute"] = minute
    out["sin_h"] = np.sin(2 * np.pi * hour / 24)
    out["cos_h"] = np.cos(2 * np.pi * hour / 24)
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering lives here so predict_proba() reproduces it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    # numeric passthrough
    for c in num_cols:
        X[c] = df[c].astype(float)
    # engineered time features
    tf = _add_time_features(df)
    for c in tf.columns:
        X[c] = tf[c]
    # categoricals for native handling
    for c in cat_cols:
        X["cat_" + c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen -> NaN
    # target encodings
    for c in cat_cols:
        m = te_maps[c]
        X["te_" + c] = df[c].map(m).fillna(m["__global__"]).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


_fit_target_encodings(train, to_y(train))

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_alpha=0,
    reg_lambda=1,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=30,
    eval_metric="auc",
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
