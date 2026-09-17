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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
N_FOLDS = 5

CFG = dict(
    n_estimators=4000,
    max_depth=16,
    learning_rate=0.07,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=10,
    reg_lambda=2.0,
    early_stopping_rounds=100,
)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
RAW_FEATURES = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in RAW_FEATURES if train[c].dtype == object]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in obj_cols}
NUMERIC_COLS = ["DepTime", "Distance"]


def _hour(df: pd.DataFrame) -> np.ndarray:
    return np.minimum((df["DepTime"].astype("int64") // 100), 24)


hour_levels = pd.Index(sorted(pd.unique(_hour(train))))


def add_numeric_feats(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    dt = df["DepTime"].astype("int64")
    hour = (dt // 100) % 24
    minute = dt % 100
    frac = (hour * 60 + minute) / 1440.0
    X["dep_hour"] = hour.astype("int64")
    X["dep_minute"] = minute.astype("int64")
    X["dep_sin"] = np.sin(2 * np.pi * frac)
    X["dep_cos"] = np.cos(2 * np.pi * frac)
    X["log_dist"] = np.log1p(df["Distance"].astype("float64"))
    X["dep_hour_cat"] = pd.Categorical(
        pd.Series(_hour(df), index=df.index), categories=hour_levels
    )
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in obj_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in NUMERIC_COLS:
        X[c] = df[c].astype("float64")
    X = add_numeric_feats(df, X)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)


def fold_params(fold: int) -> dict:
    p = dict(CFG)
    p["random_state"] = SEED + fold
    return p


t0 = time.time()
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(Xtr))
oof = np.zeros(len(Xtr))
best_iters = []
for f, (tr_idx, va_idx) in enumerate(folds):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, **fold_params(f))
    m.fit(Xtr.iloc[tr_idx], ytr[tr_idx], eval_set=[(Xtr.iloc[va_idx], ytr[va_idx])], verbose=False)
    oof[va_idx] = m.predict_proba(Xtr.iloc[va_idx])[:, 1]
    best_iters.append(int(m.best_iteration))
    print(f"fold {f}: best_iter={m.best_iteration} oof_auc={roc_auc_score(ytr[va_idx], oof[va_idx]):.4f}")
print(f"OOF AUC: {roc_auc_score(ytr, oof):.4f}")

# refit on the full train set with the iteration count chosen by CV
FIXED_ITERS = int(np.median(best_iters))
print(f"refit with n_estimators={FIXED_ITERS}")
models = []
for f in range(N_FOLDS):
    params = fold_params(f)
    params.pop("early_stopping_rounds", None)
    params["n_estimators"] = FIXED_ITERS
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS, **params)
    m.fit(Xtr, ytr)
    models.append(m)
eval_preds = np.zeros(len(Xev))
for m in models:
    eval_preds += m.predict_proba(Xev)[:, 1] / N_FOLDS
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1] / len(models)
    return p


eval_auc = roc_auc_score(yev, eval_preds)
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
