"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

CAT_COLS = ["carrier", "origin", "dest", "route"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here (predict_proba calls this on unseen rows)."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].to_numpy()
    hour = dep // 100
    minute = dep % 100
    mod = hour * 60 + minute  # minutes since midnight (values >24h stay as-is)
    X["hour"] = hour
    X["minute_of_day"] = mod
    X["sin_day"] = np.sin(2 * np.pi * mod / 1440)
    X["cos_day"] = np.cos(2 * np.pi * mod / 1440)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = df[c].str.split("-").str[-1].astype(int)
    X["distance"] = df["Distance"].to_numpy()
    X["log_distance"] = np.log1p(df["Distance"].to_numpy())
    X["carrier"] = df["UniqueCarrier"].to_numpy()
    X["origin"] = df["Origin"].to_numpy()
    X["dest"] = df["Dest"].to_numpy()
    X["route"] = (df["Origin"] + "_" + df["Dest"]).to_numpy()
    return X


# categorical levels: fitted on TRAIN ONLY
_feat_train = add_features(train)
cat_levels = {c: pd.Index(sorted(_feat_train[c].astype(str).unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = add_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])  # unseen -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))])
print(f"Training time: {time.time() - t0:.1f}s  best_iter={getattr(model, 'best_iteration', None)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
