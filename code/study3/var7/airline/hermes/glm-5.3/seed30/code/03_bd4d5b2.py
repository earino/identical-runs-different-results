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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["dep_frac"] = X["hour"] + X["minute"] / 60.0
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_frac"] / 24)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_frac"] / 24)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# hold out a validation slice of train for early stopping (last 15% of rows)
n_tr = len(train)
n_val = n_tr // 7
tr_idx = train.index[: n_tr - n_val]
va_idx = train.index[n_tr - n_val :]

X_all = prepare(train)
y_all = to_y(train)

model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=60,
)

t0 = time.time()
model.fit(X_all.iloc[tr_idx], y_all[tr_idx], eval_set=[(X_all.iloc[va_idx], y_all[va_idx])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iter: {model.best_iteration}")

FULL_MODEL_PARAMS = dict(model.get_params())
FULL_MODEL_PARAMS["n_estimators"] = max(model.best_iteration, 100) if model.best_iteration else 1500
# retrain on all rows with the early-stopped iteration count
final_model = xgb.XGBClassifier(**{**model.get_params(), "n_estimators": int(getattr(model, "best_iteration", 1500) or 1500), "early_stopping_rounds": None})
final_model.fit(X_all, y_all, verbose=False)
model = final_model
print(f"Retrain time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
