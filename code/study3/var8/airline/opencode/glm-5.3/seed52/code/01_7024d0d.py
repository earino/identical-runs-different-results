"""XGBoost binary classifier for airline delay. Feature engineering + early stopping.

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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
cat_levels["Route"] = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


def _num_from_c(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _num_from_c(df["Month"])
    day = _num_from_c(df["DayofMonth"])
    dow = _num_from_c(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    dep = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    hour = dep // 100
    tmin = hour * 60 + dep % 100
    X["hour"] = hour
    X["tmin"] = tmin
    X["tmin_sin"] = np.sin(2 * np.pi * tmin / 1440.0)
    X["tmin_cos"] = np.cos(2 * np.pi * tmin / 1440.0)
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["log_dist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce").astype(float))
    for c in ["UniqueCarrier", "Origin", "Dest"]:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=cat_levels["Route"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=1500,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=N_JOBS,
    early_stopping_rounds=50,
    eval_metric="auc",
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best_iteration: {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
