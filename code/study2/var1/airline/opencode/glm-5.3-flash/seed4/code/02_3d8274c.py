"""XGBoost binary classifier on the airline dataset (see program.md).

Contract:
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
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# Categorical levels are fitted on TRAINING data only.
CAT_LEVELS = {
    c: pd.Index(sorted(train[c].dropna().unique()))
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
}
ROUTE_LEVELS = pd.Index(
    sorted((train["Origin"] + "_" + train["Dest"]).dropna().unique())
)


def _cat(series: pd.Series, levels: pd.Index) -> pd.Categorical:
    return pd.Categorical(series, categories=levels)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["Month"] = _cat(df["Month"], CAT_LEVELS["Month"])
    X["DayofMonth"] = _cat(df["DayofMonth"], CAT_LEVELS["DayofMonth"])
    X["DayOfWeek"] = _cat(df["DayOfWeek"], CAT_LEVELS["DayOfWeek"])
    X["UniqueCarrier"] = _cat(df["UniqueCarrier"], CAT_LEVELS["UniqueCarrier"])
    X["Origin"] = _cat(df["Origin"], CAT_LEVELS["Origin"])
    X["Dest"] = _cat(df["Dest"], CAT_LEVELS["Dest"])
    X["Route"] = _cat(df["Origin"] + "_" + df["Dest"], ROUTE_LEVELS)

    dt = df["DepTime"].astype("int64")
    X["DepTime"] = dt
    hour = dt // 100
    minute = dt % 100
    tod = ((hour % 24) * 60 + minute).to_numpy(dtype=float)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["hour"] = pd.Categorical(
        hour.clip(0, 23).astype(str), categories=[str(h) for h in range(24)]
    )
    dow = df["DayOfWeek"].str.extract(r"c-(\d+)")[0].astype(float)
    X["is_weekend"] = (dow >= 6).astype("float32")
    X["Distance"] = df["Distance"].astype("float64")
    X["log_distance"] = np.log1p(df["Distance"].astype("float64"))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
EARLY_STOP = 50
VAL_FRAC = 0.15

t0 = time.time()
y_all = to_y(train)
tr_idx, va_idx = train_test_split(
    np.arange(len(train)), test_size=VAL_FRAC, random_state=SEED, stratify=y_all
)
X_tr, y_tr = prepare(train.iloc[tr_idx]), y_all[tr_idx]
X_va, y_va = prepare(train.iloc[va_idx]), y_all[va_idx]

stopper = xgb.XGBClassifier(**PARAMS, early_stopping_rounds=EARLY_STOP, eval_metric="auc")
stopper.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_iter = int(stopper.best_iteration) + 1
print(f"best_iteration: {best_iter}")

model = xgb.XGBClassifier(**{**PARAMS, "n_estimators": best_iter})
model.fit(prepare(train), y_all)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
