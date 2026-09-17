"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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

# --- feature engineering -------------------------------------------------------
C_CAT = ["UniqueCarrier", "Origin", "Dest"]          # low/medium cardinality strings
C_NUM = ["Distance", "DepTime"]
NUM_FROM_CAT = {"Month": 12, "DayofMonth": 31, "DayOfWeek": 7}  # c-<n> -> n

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in C_CAT}
route_levels = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here; predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c, mx in NUM_FROM_CAT.items():
        v = df[c].astype("string").str.replace("c-", "", regex=False)
        X[c] = pd.to_numeric(v, errors="coerce").fillna(0).astype(int) / mx
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["hour"] = hour / 23.0
    X["min_of_day"] = (hour * 60 + minute) / 1439.0
    for c in C_NUM:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["Distance"] = np.log1p(X["Distance"].clip(lower=0))
    for c in C_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str),
                                categories=route_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


FEATS = prepare(train.head(2)).columns.tolist()

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=1,
    subsample=1.0,
    colsample_bytree=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train),
          eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
