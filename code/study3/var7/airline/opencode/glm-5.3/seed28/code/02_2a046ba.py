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
# Month/DayofMonth dropped: their train(2005)->eval(2006) delay-rate profiles shift strongly
# (pure shift noise). Keep the year-stable signals: time-of-day (fine + coarse), carrier, airports, DoW.
KEEP = ["DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in
              ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
hour_levels = pd.Index(sorted((train["DepTime"] // 100).unique()))
block_levels = pd.Index(sorted((train["DepTime"] // 15 * 15).unique()))
hcar_levels = pd.Index(sorted(((train["DepTime"] // 100).astype(str) + "_" + train["UniqueCarrier"]).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[KEEP].copy()
    for c in ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    X["Hour"] = pd.Categorical(hour, categories=hour_levels)
    X["Minute"] = dep % 100
    X["Block15"] = pd.Categorical(dep // 15 * 15, categories=block_levels)
    X["HourCar"] = pd.Categorical(hour.astype(str) + "_" + df["UniqueCarrier"].astype(str), categories=hcar_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# The 2005->2006 shift punishes capacity: shallow trees + strong L1 generalize best.
model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=1,
    reg_alpha=4.0,
    reg_lambda=1.0,
    colsample_bytree=0.85,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
