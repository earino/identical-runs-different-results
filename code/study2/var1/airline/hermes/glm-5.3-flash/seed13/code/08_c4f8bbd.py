"""Airline delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Strategy notes (informed by train/eval comparisons):
- Eval (2006) has a different carrier/month delay-rate profile than train (2005): heavy
  year-specific interactions (carrier x month, route, DoM) overfit the drift. Keep the feature
  set modest and stable (weekday, carrier, airport, time-of-day), spend capacity on a deeper,
  strongly-regularized XGBoost, and average seeds/configs for stability.
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

CC = ["Month", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# encoders/statistics are fitted on training data only (never on the dataframe passed to prepare)
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CC}
DEP_BINS = pd.Index(range(48))
HOUR_BINS = pd.Index(range(24))


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
    X["dephour"] = pd.Categorical((dt // 100) % 24, categories=HOUR_BINS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# config-blend bag: alternating depth-9/gamma-3 and depth-10/gamma-5 members, 5 seeds each
CONFIGS = [
    dict(max_depth=9, gamma=3.0),
    dict(max_depth=10, gamma=5.0),
]
N_BAG = 5
BASE_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)


def make_model(seed: int, **cfg) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(random_state=seed, **BASE_PARAMS, **cfg)


t0 = time.time()
y = to_y(train)
X = prepare(train)
models = []
for cfg in CONFIGS:
    for k in range(N_BAG):
        m = make_model(seed=SEED + k, **cfg)
        m.fit(X, y)
        models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  members={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
