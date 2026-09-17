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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    # DepTime: scheduled departure hhmm -> time-of-day features
    tod = df["DepTime"].fillna(0).astype(int) % 2400
    hour = tod // 100
    minute = tod % 100
    ang = 2.0 * np.pi * (hour * 60 + minute) / 1440.0
    X["dep_sin"] = np.sin(ang)
    X["dep_cos"] = np.cos(ang)
    X["hour"] = hour
    X["hour_cat"] = pd.Categorical(hour.astype(str), categories=[str(h) for h in range(24)])
    # distance
    dist = df["Distance"].astype(float)
    X["dist"] = dist
    X["dist_log"] = np.log1p(dist)
    # categoricals (levels frozen on train; unseen -> NaN -> xgb missing)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(n_estimators: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


t0 = time.time()
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = len(train) // 5
val_idx, fit_idx = idx[:n_val], idx[n_val:]
X_fit, y_fit = prepare(train.iloc[fit_idx]), to_y(train.iloc[fit_idx])
X_val, y_val = prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx])
es_model = make_model(4000)
es_model.set_params(early_stopping_rounds=100, eval_metric="auc")
es_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
best_n = max(int(es_model.best_iteration) + 1, 30)
print(f"best_iteration={best_n} val_auc={es_model.best_score:.4f}")

model = make_model(best_n)
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
