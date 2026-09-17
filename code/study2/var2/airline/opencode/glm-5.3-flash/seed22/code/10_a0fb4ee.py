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
cat_cols = [c for c in obj_cols if train[c].nunique() <= 5000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
cat_levels["Hour"] = pd.Index(range(24))
cat_levels["Tod"] = pd.Index(["n1", "n2", "am", "day", "pm", "eve", "late"])
cat_levels["Route"] = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    hour = (df["DepTime"] // 100) % 24
    t = hour * 60 + df["DepTime"] % 100
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["Hour"] = pd.Categorical(hour, categories=cat_levels["Hour"])
    X["Tod"] = pd.Categorical(pd.cut(t, bins=[-1, 359, 419, 599, 959, 1139, 1259, 1439], labels=["n1", "n2", "am", "day", "pm", "eve", "late"]), categories=cat_levels["Tod"])
    X["Minute"] = df["DepTime"] % 100
    X["SinT"] = np.sin(2 * np.pi * t / 1440.0)
    X["CosT"] = np.cos(2 * np.pi * t / 1440.0)
    X["LogDist"] = np.log1p(df["Distance"])
    X["Route"] = pd.Categorical(df["Origin"] + "_" + df["Dest"], categories=cat_levels["Route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=300,
    max_depth=14,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=0.8,
    tree_method="hist",
    max_bin=512,
    enable_categorical=True,
    n_jobs=N_JOBS,
)
CONFIGS = [dict(reg_alpha=6), dict(reg_alpha=6), dict(reg_alpha=6), dict(reg_alpha=10), dict(reg_alpha=10), dict(reg_alpha=10)]
SEEDS = [1, 2, 3, 1, 2, 3]

t0 = time.time()
X = prepare(train)
y = to_y(train)
models = [xgb.XGBClassifier(random_state=s, **{**PARAMS, **cfg}).fit(X, y) for s, cfg in zip(SEEDS, CONFIGS)]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
