"""XGBoost binary classifier for the airline delay task. THIS IS THE ONLY FILE THE AGENT EDITS.

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
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# interaction categorical fitted on TRAINING data only
_carrier_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + (train["DepTime"].astype("int32") // 100).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function is NOT applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = df["DepTime"].astype("int32")
    hr = (dt // 100) % 24
    mn = dt % 100
    hf = hr + mn / 60.0
    X["dep_hour"] = hr.astype("float32")
    X["dep_min"] = mn.astype("float32")
    X["hour_sin"] = np.sin(2 * np.pi * hf / 24).astype("float32")
    X["hour_cos"] = np.cos(2 * np.pi * hf / 24).astype("float32")
    X["hour_sq"] = (hf ** 2).astype("float32")
    X["is_night"] = ((hr >= 0) & (hr <= 5)).astype("int8")
    X["is_evening"] = ((hr >= 17) & (hr <= 23)).astype("int8")
    X["is_midday"] = ((hr >= 10) & (hr <= 16)).astype("int8")
    ch = df["UniqueCarrier"].astype(str) + "_" + hr.astype(str)
    X["carrier_hour"] = pd.Categorical(ch, categories=_carrier_hour_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    eval_metric="auc",
    max_bin=512,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)
models = []
for md, seed in [(8, 42), (10, 7), (8, 7)]:
    m = xgb.XGBClassifier(max_depth=md, random_state=seed, **PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    print(f"depth={md} best_iter={m.best_iteration}")
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    # xgboost >= 2.0: with early stopping, predict automatically uses best_iteration
    Ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
