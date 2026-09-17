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
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}
hc_levels = pd.Index(sorted((train["UniqueCarrier"] + "_" + (train["DepTime"] // 100).astype(str)).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in ["Month", "DayofMonth", "DayOfWeek"]:
        X[c] = df[c].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    mins = (dep // 100) * 60 + dep % 100
    X["DepHour"] = dep // 100
    X["DepMins"] = mins
    X["DepSin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["DepCos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["Distance"] = df["Distance"]
    X["DistLog"] = np.log1p(df["Distance"].astype(float))
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    hour_s = (df["DepTime"] // 100).astype(str)
    X["HourCarrier"] = pd.Categorical(df["UniqueCarrier"] + "_" + hour_s, categories=hc_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 3-seed ensemble of deep XGB models (variance reduction) ----------------
ENSEMBLE_PARAMS = dict(
    n_estimators=200,
    max_depth=20,
    learning_rate=0.02,
    subsample=1.0,
    colsample_bytree=0.4,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
models = []
t0 = time.time()
for seed in (1, 2, 3):
    m = xgb.XGBClassifier(random_state=seed, **ENSEMBLE_PARAMS)
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
