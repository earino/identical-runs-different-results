"""XGBoost binary classifier on the airline delay dataset.

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

# category levels fit on TRAIN only (never on the df passed to predict_proba)
cat_levels = {
    "Carrier": pd.Index(sorted(train["UniqueCarrier"].astype(str).unique())),
    "Origin": pd.Index(sorted(train["Origin"].astype(str).unique())),
    "Dest": pd.Index(sorted(train["Dest"].astype(str).unique())),
    "Hour": pd.Index(list(range(0, 27))),
}

# carrier x hour levels fit on TRAIN only
_carrier_tr = train["UniqueCarrier"].astype(str)
_hour_tr = (train["DepTime"].astype(np.int64) // 100).astype(str)
cat_levels["CarrierHour"] = pd.Index(sorted(set(_carrier_tr + "|" + _hour_tr)))

def fe(df: pd.DataFrame) -> pd.DataFrame:
    """All feature engineering lives here; predict_proba() calls it on unseen rows."""
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayOfMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    dt = df["DepTime"].astype(np.int64)
    hh, mm = dt // 100, dt % 100
    X["DepTime"] = dt
    X["Hour"] = hh
    X["HourCat"] = pd.Categorical(hh, categories=cat_levels["Hour"])
    X["CarrierHour"] = pd.Categorical(
        df["UniqueCarrier"].astype(str) + "|" + hh.astype(str), categories=cat_levels["CarrierHour"]
    )
    X["Minute"] = mm
    mod = ((hh * 60 + mm) % 1440).to_numpy(dtype=float)
    X["tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    dist = df["Distance"].to_numpy(dtype=float)
    X["Distance"] = dist
    X["logDist"] = np.log1p(dist)
    X["Carrier"] = pd.Categorical(df["UniqueCarrier"].astype(str), categories=cat_levels["Carrier"])
    X["Origin"] = pd.Categorical(df["Origin"].astype(str), categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"].astype(str), categories=cat_levels["Dest"])
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return fe(df)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


model = xgb.XGBClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=50,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_tr = prepare(train)
model.fit(X_tr, to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
