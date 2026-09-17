"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives inside prepare(); it is fitted on the training data only.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- static feature metadata (fitted on train only) -----------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}
HOUR_LEVELS = [str(i) for i in range(24)]
QHOUR_LEVELS = [f"{h}-{q}" for h in range(24) for q in range(4)]
CARRIER_HOUR_LEVELS = sorted(
    {f"{c}|{h}" for c in train["UniqueCarrier"].astype(str) for h in range(24)}
)
N_MODELS = 3
N_TREES = 300
LR = 0.04


def _parse_c(s: pd.Series) -> pd.Series:
    """c-7 -> 7"""
    return s.astype(str).str.split("-").str[-1].astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _parse_c(df["Month"])
    X["DayofMonth"] = _parse_c(df["DayofMonth"])
    X["DayOfWeek"] = _parse_c(df["DayOfWeek"])

    dep = df["DepTime"].astype(int).clip(0, 2359)
    hour = dep // 100
    minute = dep % 100
    X["DepHour"] = hour
    X["DepMinute"] = minute
    minofday = hour * 60 + minute
    X["MinOfDay"] = minofday
    X["HourSin"] = np.sin(2 * np.pi * minofday / 1440.0)
    X["HourCos"] = np.cos(2 * np.pi * minofday / 1440.0)
    X["DepTimeRaw"] = dep.astype(float)
    X["DepHourC"] = pd.Categorical(hour.astype(str), categories=HOUR_LEVELS)
    X["QHourC"] = pd.Categorical(
        hour.astype(str) + "-" + (minute // 15).astype(str), categories=QHOUR_LEVELS
    )

    X["Distance"] = df["Distance"].astype(float)
    X["LogDistance"] = np.log1p(X["Distance"])

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["CarrierHour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "|" + hour.astype(str),
        categories=CARRIER_HOUR_LEVELS,
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: 5-seed ensemble of deep, strongly regularized trees ------------------
X_tr, y_tr = prepare(train), to_y(train)

MODELS = []
t0 = time.time()
for s in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=N_TREES,
        learning_rate=LR,
        max_depth=16,
        reg_alpha=1.0,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        enable_categorical=True,
        random_state=s,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_tr)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
