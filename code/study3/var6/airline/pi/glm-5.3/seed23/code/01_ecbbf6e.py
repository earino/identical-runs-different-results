"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives inside prepare() so it applies to the hidden holdout too.
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


# --- fitted on train only, reused by prepare() for any dataframe ----------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
_cat_train = pd.DataFrame({
    "UniqueCarrier": train["UniqueCarrier"],
    "Origin": train["Origin"],
    "Dest": train["Dest"],
    "Route": train["Origin"].astype(str) + "_" + train["Dest"].astype(str),
})
CAT_LEVELS = {c: pd.Index(sorted(_cat_train[c].dropna().unique())) for c in CAT_COLS}


def _cnum(s: pd.Series) -> pd.Series:
    """'c-7' -> 7 (returns float so NaN stays NaN)."""
    return s.astype(str).str.extract(r"(\d+)")[0].astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(float)
    hour = (dep // 100).clip(0, 24)
    minute = dep % 100
    tod = hour + minute / 60.0
    X["dep_time_raw"] = dep
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    X["doy"] = (month - 1) * 31 + day
    X["distance"] = df["Distance"].astype(float)
    X["log_distance"] = np.log1p(X["distance"])
    X["Route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    X["UniqueCarrier"] = df["UniqueCarrier"].values
    X["Origin"] = df["Origin"].values
    X["Dest"] = df["Dest"].values
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    eval_metric="auc",
    early_stopping_rounds=50,
)

t0 = time.time()
model.fit(prepare(train), to_y(train), eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
