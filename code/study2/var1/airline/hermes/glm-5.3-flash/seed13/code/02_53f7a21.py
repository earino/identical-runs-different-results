"""Airline delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Strategy notes (informed by train/eval comparisons):
- Eval (2006) has a different carrier/month delay-rate profile than train (2005), so heavyweight
  interactions on unstable effects hurt eval. Keep feature set modest (day-level categorical +
  departure time-of-day) and spend capacity on a deeper/lower-rate tuned XGBoost.
- All encoders/statistics are fitted on train only; prepare() is the single feature path.
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

CC = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# encoders/statistics are fitted on training data only (never on the dataframe passed to prepare)
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CC}
ROUTE_LEVELS = pd.Index(sorted((train["Origin"] + "_" + train["Dest"]).unique()))
ORIGIN_COUNTS = train["Origin"].value_counts()
DEST_COUNTS = train["Dest"].value_counts()
DEP_BINS = pd.Index(range(48))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CC:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    dt = df["DepTime"].astype(int)
    mins = (dt // 100) * 60 + (dt % 100)
    X["depbin"] = pd.Categorical((mins % 1440) // 30, categories=DEP_BINS)
    X["route"] = pd.Categorical(df["Origin"] + "_" + df["Dest"], categories=ROUTE_LEVELS)
    X["origin_n"] = df["Origin"].map(ORIGIN_COUNTS).astype(float)
    X["dest_n"] = df["Dest"].map(DEST_COUNTS).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed: int = SEED) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.5,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
y = to_y(train)
X = prepare(train)
model = make_model()
model.fit(X, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
