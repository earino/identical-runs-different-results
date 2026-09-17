"""XGBoost airline delay classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
BASE_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in BASE_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
BASE_COLS = [c for c in BASE_COLS if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[BASE_COLS].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # cyclical departure time (DepTime is hhmm; values can exceed 2400 near midnight)
    dm = df["DepTime"] % 2400
    minutes = (dm // 100) * 60 + (dm % 100)
    X["dep_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    X["dep_hour"] = df["DepTime"] // 100  # linear hour bucket
    X["dep_minute"] = minutes
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of deep, heavily column-subsampled XGB models ------------
# Under the 2005->2006 shift, deep trees with colsample 0.4 generalize far better
# than anything shallow; averaging decorrelated deep models adds ~+0.001.
PARAMS = dict(
    n_estimators=200,
    max_depth=24,
    learning_rate=0.06,
    tree_method="hist",
    enable_categorical=True,
    colsample_bytree=0.4,
    max_bin=512,
    random_state=SEED,
    n_jobs=N_JOBS,
)
DEPTHS = [20, 22, 24, 26, 28]

t0 = time.time()
X_tr, y_tr = prepare(train), to_y(train)
models = []
for d in DEPTHS:
    m = xgb.XGBClassifier(**{**PARAMS, "max_depth": d, "random_state": d})
    m.fit(X_tr, y_tr, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
