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
freq_cols = ["Origin", "Dest", "route"]
feature_cols = feature_cols + ["minutes_of_day", "hour", "minute", "log_distance"] \
               + [c + "_freq" for c in freq_cols]
num_extra = ["minutes_of_day", "hour", "minute", "log_distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# frequency statistics from the training data only (label-free, contract)
_route_series = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_freq_maps = {}
for name, keys in [("Origin", train["Origin"]), ("Dest", train["Dest"]), ("route", _route_series)]:
    vc = keys.value_counts()
    _freq_maps[name] = (np.log1p(vc)).to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df.copy()
    dep = df["DepTime"]
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    X["minutes_of_day"] = (X["hour"] * 60 + X["minute"]).astype(float)
    X["route"] = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    X["log_distance"] = np.log1p(X["Distance"].astype(float))
    for name in freq_cols:
        X[name + "_freq"] = X[name].map(_freq_maps[name]).fillna(0.0)
    X = X[feature_cols]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 5
X_tr, y_tr = prepare(train), to_y(train)
X_ev, y_ev = prepare(evald), to_y(evald)

t0 = time.time()
models = []
for seed in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=60,
        eval_metric="auc",
        random_state=SEED + seed,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    models.append(m)
    print(f"  model {seed}: best_round={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
