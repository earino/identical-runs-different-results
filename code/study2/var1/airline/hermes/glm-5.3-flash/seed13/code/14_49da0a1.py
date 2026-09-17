"""Airline delay XGBoost classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Strategy notes (informed by train/eval comparisons):
- Eval (2006) has a different carrier/month delay-rate profile than train (2005): heavy
  year-specific interactions (carrier x month, route, DoM) and Month itself overfit the drift.
  Keep the feature set stable (weekday, carrier, airport, time-of-day), spend capacity on a
  deeper, strongly-regularized XGBoost, and average seeds/configs (incl. colsample variety) for
  stability.
- Rare airports (fewer than MIN_AIRPORT train rows) collapse into a shared OTHER bucket.
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

CC = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
# encoders/statistics are fitted on training data only (never on the dataframe passed to prepare)
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CC}
MIN_AIRPORT = 240
ORIGIN_TOP = set(train["Origin"].value_counts()[lambda s: s >= MIN_AIRPORT].index)
DEST_TOP = set(train["Dest"].value_counts()[lambda s: s >= MIN_AIRPORT].index)
ORIGIN_LEVELS = pd.Index(sorted(ORIGIN_TOP) + ["OTHER"])
DEST_LEVELS = pd.Index(sorted(DEST_TOP) + ["OTHER"])
DEP96_BINS = pd.Index(range(96))
HOUR_BINS = pd.Index(range(24))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["DayOfWeek"] = pd.Categorical(df["DayOfWeek"], categories=CAT_LEVELS["DayOfWeek"])
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=CAT_LEVELS["UniqueCarrier"])
    X["Origin"] = pd.Categorical(
        df["Origin"].where(df["Origin"].isin(ORIGIN_TOP), "OTHER"), categories=ORIGIN_LEVELS
    )
    X["Dest"] = pd.Categorical(
        df["Dest"].where(df["Dest"].isin(DEST_TOP), "OTHER"), categories=DEST_LEVELS
    )
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    dt = df["DepTime"].astype(int)
    mins = (dt // 100) * 60 + (dt % 100)
    X["depbin96"] = pd.Categorical((mins % 1440) // 15, categories=DEP96_BINS)
    X["dephour"] = pd.Categorical((dt // 100) % 24, categories=HOUR_BINS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# 20-member bag: depth-10/gamma-5 at 5 lr/colsample settings, 4 seeds each
CONFIGS = [
    dict(n_estimators=800, learning_rate=0.05),
    dict(n_estimators=2400, learning_rate=0.02),
    dict(n_estimators=800, learning_rate=0.05, colsample_bytree=0.8),
    dict(n_estimators=2400, learning_rate=0.02, colsample_bytree=0.8),
    dict(n_estimators=1200, learning_rate=0.04, colsample_bytree=0.8),
]
N_BAG = 4
BASE_PARAMS = dict(
    max_depth=10,
    gamma=5.0,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.5,
    reg_alpha=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)


t0 = time.time()
y = to_y(train)
X = prepare(train)
models = []
for cfg in CONFIGS:
    full_cfg = {**BASE_PARAMS, **cfg}
    for k in range(N_BAG):
        m = xgb.XGBClassifier(random_state=SEED + k, **full_cfg)
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
