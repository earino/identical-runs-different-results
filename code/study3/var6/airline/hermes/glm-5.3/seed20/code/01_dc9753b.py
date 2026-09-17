"""XGBoost binary classifier for airline delay prediction.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives in prepare(df); statistics/categories are fit on train only.
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

# --- feature engineering -------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "hour_route", "hour_origin"]


def _cnum(s: pd.Series) -> pd.Series:
    """c-7 -> 7 (robust to plain numbers too)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    # scheduled departure time (hhmm int)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).clip(0, 26)
    X["dep_time"] = dt
    X["dep_hour"] = hour
    X["dep_sin"] = np.sin(2 * np.pi * dt / 2400.0)
    X["dep_cos"] = np.cos(2 * np.pi * dt / 2400.0)
    # distance
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_distance"] = np.log1p(dist)
    # categoricals (route/carrier/location)
    X["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["hour_route"] = hour.astype(str) + "_" + X["route"]
    X["hour_origin"] = hour.astype(str) + "_" + df["Origin"].astype(str)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    return X


_base_train = _base_features(train)
cat_levels = {c: pd.Index(sorted(_base_train[c].unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = _base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)
X_tr, X_val, y_val_ = None, None, None
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=SEED, stratify=y_all)

model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=20,
    early_stopping_rounds=40,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
