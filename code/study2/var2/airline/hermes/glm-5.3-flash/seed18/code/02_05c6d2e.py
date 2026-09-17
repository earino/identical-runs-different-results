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
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # decode c-<n> strings to integer ordinals
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = df[c].astype(str).str.replace("c-", "", regex=False).astype(float)
    # scheduled departure time: hour-of-day is the dominant delay driver; add cyclical encoding
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).fillna(0.0)
    minute = (dt % 100).fillna(0.0)
    X["DepHour"] = hour
    ang = 2.0 * np.pi * (hour * 60.0 + minute) / 1440.0
    X["DepSin"] = np.sin(ang)
    X["DepCos"] = np.cos(ang)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0.0)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
    eval_metric="auc",
)

y_all = to_y(train)
X_all = prepare(train)

# time-like split: last 20% of the training file as validation for early stopping
n_val = int(0.2 * len(X_all))
X_fit, y_fit = X_all.iloc[:-n_val], y_all[: len(X_all) - n_val]
X_val, y_val = X_all.iloc[-n_val:], y_all[len(X_all) - n_val :]

t0 = time.time()
es_model = xgb.XGBClassifier(**PARAMS)
es_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
best_it = int(getattr(es_model, "best_iteration", PARAMS["n_estimators"] - 1)) + 1
print(f"Early-stop fit: {time.time() - t0:.1f}s, best_iteration={best_it}")
del es_model

# refit on ALL training rows with the chosen iteration count (+10% for the extra 25% data)
t0 = time.time()
model = xgb.XGBClassifier(**{k: v for k, v in PARAMS.items() if k not in ("early_stopping_rounds",)})
model.set_params(n_estimators=min(int(best_it * 1.1) + 1, PARAMS["n_estimators"]))
model.fit(X_all, y_all)
print(f"Full refit: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
