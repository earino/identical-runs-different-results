"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

base_features = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in base_features if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

DERIVED = ["dep_hour", "dep_tod", "dep_sin", "dep_cos", "log_dist"]
FEATURES = base_features + DERIVED


def build(df: pd.DataFrame) -> pd.DataFrame:
    X = df[base_features].copy()
    dep = df["DepTime"].astype(int).to_numpy()
    dep = np.where(dep >= 2400, dep - 2400, dep)
    hour = np.clip(dep // 100, 0, 23)
    tod = hour * 60 + np.clip(dep % 100, 0, 59)
    X["dep_hour"] = hour
    X["dep_tod"] = tod
    X["dep_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X[FEATURES]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return build(df)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=4000,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.4,
    min_child_weight=30,
    reg_lambda=5.0,
    reg_alpha=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=80,
)

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
model.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
