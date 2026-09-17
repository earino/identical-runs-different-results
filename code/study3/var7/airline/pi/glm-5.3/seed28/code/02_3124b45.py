"""XGBoost binary classifier — airline dep delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives inside prepare(), fit only on train data.
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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- fitted-on-train statistics (never on incoming df) -------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
cat_levels["Flight"] = pd.Index(sorted(_route.unique()))

DEG2RAD = np.pi / 180.0


def _int_col(s: pd.Series) -> pd.Series:
    # 'c-7' -> 7 (also tolerates plain ints)
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # calendar numerics + cyclical
    m = _int_col(df["Month"])
    dom = _int_col(df["DayofMonth"])
    dow = _int_col(df["DayOfWeek"])
    X["month"] = m
    X["dom"] = dom
    X["dow"] = dow
    X["month_sin"] = np.sin(2 * np.pi * m / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * m / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # departure time
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24)
    minute = dep % 100
    tmin = hour * 60 + minute  # minutes since midnight
    X["DepTime"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tmin"] = tmin
    X["tmin_sin"] = np.sin(2 * np.pi * tmin / 1440.0)
    X["tmin_cos"] = np.cos(2 * np.pi * tmin / 1440.0)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["log_distance"] = np.log1p(dist)
    # route
    X["Flight"] = pd.Categorical(
        (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)), categories=cat_levels["Flight"]
    )  # unseen routes -> missing
    # categoricals with train-fitted levels (unseen -> NaN/missing)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    # cyclical dom
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31.0)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31.0)
    return X


# --- model ---------------------------------------------------------------------
Xall = prepare(train)
yall = to_y(train)

model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=8,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xall, yall)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
