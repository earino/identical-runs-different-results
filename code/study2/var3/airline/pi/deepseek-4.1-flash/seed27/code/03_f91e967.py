"""XGBoost binary classifier for the airline delay task.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives in `prepare()`, which `predict_proba` uses, so it applies to any raw frame.
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
SEEDS = [42, 1, 7, 13]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering ------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
NUM_COLS = ["DepTime", "Distance", "hour", "minute"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    t = X["DepTime"].to_numpy()
    hour = (t // 100) % 24  # hhmm; >2400 means after-midnight, wrap into 0..23
    X["hour"] = hour
    X["minute"] = t % 100
    # Carrier x departure-hour interaction: strong, transfers across years.
    X["carrier_hour"] = X["UniqueCarrier"].astype(str) + "_" + pd.Series(hour, index=X.index).astype(str)
    return X


_train_d = add_derived(train)
_cat_levels = {c: pd.Index(sorted(_train_d[c].dropna().unique())) for c in CAT_COLS + ["carrier_hour"]}
FEATURES = CAT_COLS + NUM_COLS + ["carrier_hour"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = add_derived(df)[FEATURES].copy()
    for c in _cat_levels:
        X[c] = pd.Categorical(X[c], categories=_cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: seed ensemble of loss-guided XGBoost ------------------------------
def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=900,
        max_depth=0,
        grow_policy="lossguide",
        max_leaves=192,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
for seed in SEEDS:
    m = make_model(seed)
    m.fit(X_train, y_train, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
