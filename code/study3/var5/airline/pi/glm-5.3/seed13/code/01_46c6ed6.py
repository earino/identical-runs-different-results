"""XGBoost binary classifier for airline delay prediction.

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

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest", "Route"]


def add_base(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["Route"] = X["Origin"].astype(str) + "_" + X["Dest"].astype(str)
    return X


_train_base = add_base(train)
cat_levels = {c: pd.Index(sorted(_train_base[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = add_base(df)
    dt = X["DepTime"]
    hour = (dt // 100).astype(float)
    minute = (dt % 100).astype(float)
    frac_hour = hour + minute / 60.0
    X["hour"] = hour
    X["minute"] = minute
    X["frac_hour"] = frac_hour
    X["hour_sin"] = np.sin(2 * np.pi * frac_hour / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * frac_hour / 24.0)
    X["month_num"] = X["Month"].str.slice(2).astype(int)
    X["day_num"] = X["DayofMonth"].str.slice(2).astype(int)
    X["dow_num"] = X["DayOfWeek"].str.slice(2).astype(int)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow_num"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow_num"] / 7.0)
    X["month_sin"] = np.sin(2 * np.pi * X["month_num"] / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * X["month_num"] / 12.0)
    out = pd.DataFrame(index=X.index)
    num_cols = ["DepTime", "Distance", "hour", "minute", "frac_hour", "hour_sin", "hour_cos",
                "month_num", "day_num", "dow_num", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    for c in num_cols:
        out[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in CAT_COLS:
        out[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=1500,
    max_depth=8,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
rng = np.random.RandomState(SEED)
val_mask = rng.rand(len(train)) < 0.15
Xall = prepare(train)
yall = to_y(train)
fit_idx = np.where(~val_mask)[0]
val_idx = np.where(val_mask)[0]
model = xgb.XGBClassifier(**PARAMS)
model.fit(Xall.iloc[fit_idx], yall[fit_idx], eval_set=[(Xall.iloc[val_idx], yall[val_idx])], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
