"""Airline delay XGBoost — depth sweep on baseline features (env knobs)."""
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
N_TREES = int(os.environ.get("N_TREES", "60"))
DEPTH = int(os.environ.get("DEPTH", "4"))
LR = float(os.environ.get("LR", "0.1"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features (baseline) --------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
t0 = time.time()
model = xgb.XGBClassifier(
    n_estimators=N_TREES,
    max_depth=DEPTH,
    learning_rate=LR,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s n_trees={N_TREES} depth={DEPTH} lr={LR}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
