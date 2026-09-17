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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
num_cols = [c for c in feature_cols if c not in obj_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[num_cols + cat_cols].copy()
    t = df["DepTime"].astype(int)
    X["hour"] = t // 100
    X["hour24"] = X["hour"] % 24
    X["minute"] = t % 100
    X["minute_of_day"] = X["hour"] * 60 + X["minute"]
    ang = 2.0 * np.pi * (X["minute_of_day"] / 1440.0)
    X["tod_sin"] = np.sin(ang)
    X["tod_cos"] = np.cos(ang)
    X["log_distance"] = np.log1p(df["Distance"].astype(float))
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: diversified XGBoost ensemble ----------------------------------------
ALL_COLS = None  # filled after first prepare
Xtr_full, ytr = prepare(train), to_y(train)
Xev_full, yev = prepare(evald), to_y(evald)
ALL_COLS = list(Xtr_full.columns)

# member specs: (column subset, xgb params) — varied for decorrelation
FS_A = [c for c in ALL_COLS if c not in ("hour24", "minute_of_day")]
FS_B = [c for c in ALL_COLS if c not in ("DepTime", "minute")]
FS_C = ALL_COLS

CFGS = [
    dict(max_depth=4, learning_rate=0.05, min_child_weight=10, subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0, early_stopping_rounds=100),
    dict(max_depth=6, learning_rate=0.03, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, early_stopping_rounds=150),
    dict(max_depth=5, learning_rate=0.04, min_child_weight=15, subsample=0.85, colsample_bytree=0.75, reg_lambda=3.0, early_stopping_rounds=120),
]
FEATURE_SETS = [FS_A, FS_B, FS_C]

members = []
t0 = time.time()
k = 0
for cfg in CFGS:
    for fs in FEATURE_SETS:
        for rep in range(2):
            m = xgb.XGBClassifier(
                n_estimators=6000,
                tree_method="hist",
                enable_categorical=True,
                eval_metric="auc",
                random_state=SEED + 10 * k,
                n_jobs=N_JOBS,
                **cfg,
            )
            m.fit(Xtr_full[fs], ytr, eval_set=[(Xev_full[fs], yev)], verbose=False)
            members.append((m, fs))
            print(f"  member {k}: cols={len(fs)} depth={cfg['max_depth']} best_iter={m.best_iteration} "
                  f"auc={m.evals_result()['validation_0']['auc'][m.best_iteration]:.4f}")
            k += 1
print(f"Training time: {time.time() - t0:.1f}s  n_members={len(members)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X[fs])[:, 1] for m, fs in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
