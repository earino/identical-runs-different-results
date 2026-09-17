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

# --- feature engineering ------------------------------------------------------
# Everything here is computed from the raw input frame only, so prepare() reproduces it on unseen rows.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    dpm = df["DepTime"].astype("int64")
    hh = (dpm // 100) % 24
    mm = (dpm % 100).clip(0, 59)
    X = pd.DataFrame(index=df.index)
    X["hour"] = hh.astype("int32")
    X["minute"] = mm.astype("int32")
    X["mins"] = (hh * 60 + mm).astype("int32")
    X["hour_sin"] = np.sin(2 * np.pi * hh / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hh / 24.0)
    X["month"] = df["Month"].str.slice(2).astype("int32")
    X["dom"] = df["DayofMonth"].str.slice(2).astype("int32")
    X["dow"] = df["DayOfWeek"].str.slice(2).astype("int32")
    X["dow_sin"] = np.sin(2 * np.pi * X["dow"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow"] / 7.0)
    X["is_weekend"] = (X["dow"] >= 6).astype("int8")
    X["distance"] = df["Distance"].astype("float32")
    X["log_distance"] = np.log1p(X["distance"])
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    return X


TRAIN_ENG = _engineer(train)
CAT_LEVELS = {c: pd.Index(sorted(TRAIN_ENG[c].unique())) for c in CAT_COLS}
FEATURE_COLS = list(TRAIN_ENG.columns)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _engineer(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X[FEATURE_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr = prepare(train)
ytr = to_y(train)
rng = np.random.default_rng(SEED)
val_mask = rng.random(len(Xtr)) < 0.1

PARAMS = dict(
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=1,
    random_state=SEED,
    n_jobs=N_JOBS,
)

# Stage 1: probe the stopping round on a held-out slice of train (early stopping needs an eval set).
t0 = time.time()
probe = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=100, eval_metric="auc", **PARAMS)
probe.fit(
    Xtr[~val_mask],
    ytr[~val_mask],
    eval_set=[(Xtr[val_mask], ytr[val_mask])],
    verbose=False,
)
ROUND_SCALE = 0.7
best_rounds = max(1, int((int(probe.best_iteration) + 1) * ROUND_SCALE))
print(f"Probe time: {time.time() - t0:.1f}s  best_iter={best_rounds}")

# Stage 2: refit on all of train with that many rounds, bagged over seeds to cut variance.
t0 = time.time()
SEEDS = [42, 7, 2024]
models = []
for sd in SEEDS:
    m = xgb.XGBClassifier(n_estimators=best_rounds, **{**PARAMS, "random_state": sd})
    m.fit(Xtr, ytr, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  n_models={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
