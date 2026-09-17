"""XGBoost binary classifier for airline departure delay.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives inside prepare(), which only uses statistics fit on train.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# categorical string columns kept as categoricals (levels fit on TRAIN only)
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
ROUTE = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
cat_levels["Route"] = pd.Index(sorted(ROUTE.unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe -> feature matrix. Uses only train-fit statistics (cat_levels)."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    tod = hour * 60 + minute  # minute of day
    X["DepTime"] = dep
    X["tod"] = tod
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Route"] = pd.Categorical(route, categories=cat_levels["Route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.1, random_state=SEED, stratify=y_all
)

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iters: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
