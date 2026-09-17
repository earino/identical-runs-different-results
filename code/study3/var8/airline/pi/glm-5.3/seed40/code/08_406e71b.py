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
SEEDS = [42, 1, 2]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].unique())) for c in CAT_COLS}
# carrier x hour-of-day: the strongest interaction on this dataset
_htr = (train["DepTime"].astype(int) // 100).astype(str)
_ch_counts = (train["UniqueCarrier"] + "_" + _htr).value_counts()
CAR_HOUR_LEVELS = pd.Index(_ch_counts[_ch_counts >= 50].index)
# origin x hour-of-day, rare combos (fewer than 100 train flights) left as NaN
_oh_counts = (train["Origin"] + "_" + _htr).value_counts()
ORIGIN_HOUR_LEVELS = pd.Index(_oh_counts[_oh_counts >= 100].index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.slice(2).astype(int)
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    X["dep_time"] = dep
    X["minute"] = dep % 100
    # hour of day: strongest single signal, non-monotonic -> native categorical
    X["hour"] = pd.Categorical(dep // 100)
    X["distance"] = df["Distance"].astype(int)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    X["car_hour"] = pd.Categorical(
        df["UniqueCarrier"] + "_" + (dep // 100).astype(str), categories=CAR_HOUR_LEVELS
    )
    X["origin_hour"] = pd.Categorical(
        df["Origin"] + "_" + (dep // 100).astype(str), categories=ORIGIN_HOUR_LEVELS
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# 2005->2006 time shift: keep shrinkage high and subsample rows/features.
# Deep trees + the carrier-hour interaction generalise best; average 5 seeds.
def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=16,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
X_train, y_train = prepare(train), to_y(train)
models = []
for seed in SEEDS:
    m = make_model(seed)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
