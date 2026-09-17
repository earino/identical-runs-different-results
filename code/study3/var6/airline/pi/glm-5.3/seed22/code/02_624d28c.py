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
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}
route_levels = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = df[c].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    hour = dep // 100
    minute = dep % 100
    mins = hour * 60 + minute
    X["DepHour"] = hour
    X["DepMin"] = minute
    X["DepMins"] = mins
    X["DepSin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["DepCos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["Distance"] = df["Distance"]
    X["DistLog"] = np.log1p(df["Distance"].astype(float))
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    route = df["Origin"] + "_" + df["Dest"]
    X["Route"] = pd.Categorical(route, categories=route_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr, Xva, ytr, yva = train_test_split(prepare(train), to_y(train), test_size=0.1, random_state=SEED, stratify=to_y(train))
model.fit(Xtr, ytr, eval_set=[(Xva, yva)])
print(f"best iter: {model.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
