"""XGBoost binary classifier for airline delay prediction.

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
SEED = 4242

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# 15-min departure-time bands seen in train (fit on train only)
_train_band = (train["DepTime"] // 100 * 4 + (train["DepTime"] % 100) // 15)
dep_band_levels = pd.Index(sorted(_train_band.unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["dep_frac"] = X["hour"] + X["minute"] / 60.0
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_frac"] / 24)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_frac"] / 24)
    # 15-minute departure-time band as a categorical
    band = (X["DepTime"] // 100 * 4 + X["minute"] // 15).astype(int)
    X["dep_band"] = pd.Categorical(band, categories=dep_band_levels)
    # day-of-year and month-day interaction numerics
    m = X["Month"].str[2:].astype(int)
    d = X["DayofMonth"].str[2:].astype(int)
    X["month"] = m
    X["day"] = d
    X["dow"] = X["DayOfWeek"].str[2:].astype(int)
    X["doy"] = (m - 1) * 31 + d  # monotone proxy for day-of-year
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# bagged ensemble of XGB models: 3 seeds on 90% row samples, average probabilities
X_all = prepare(train)
y_all = to_y(train)
n_tr = len(train)
rng = np.random.RandomState(123)
BAG = 3
MODELS = []
t0 = time.time()
for b in range(BAG):
    idx = rng.choice(n_tr, size=int(n_tr * 0.9), replace=False)
    m = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + b,
        n_jobs=N_JOBS,
    )
    m.fit(X_all.iloc[idx], y_all[idx], verbose=False)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({BAG} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = np.column_stack([m.predict_proba(X)[:, 1] for m in MODELS])
    return ps.mean(axis=1)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
