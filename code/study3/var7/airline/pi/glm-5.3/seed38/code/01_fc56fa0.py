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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- static feature metadata (fitted on train only) -----------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
route_train = train["Origin"].astype(str) + "-" + train["Dest"].astype(str)
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}
cat_levels["Route"] = pd.Index(sorted(route_train.unique()))
ROUTE_LEVELS = cat_levels["Route"]


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

    X["Distance"] = df["Distance"].astype(float)
    X["LogDistance"] = np.log1p(X["Distance"])

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Route"] = pd.Categorical(
        df["Origin"].astype(str) + "-" + df["Dest"].astype(str), categories=ROUTE_LEVELS
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ---------------------------------------------------------------------
y = to_y(train)
X_all = prepare(train)
X_tr, X_va, y_tr, y_va = train_test_split(X_all, y, test_size=0.1, random_state=SEED, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=1500,
    learning_rate=0.1,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s, best iter {model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
