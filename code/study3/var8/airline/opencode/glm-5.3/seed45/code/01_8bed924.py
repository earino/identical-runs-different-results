"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# raw feature columns used by prepare(); engineered columns are derived from these
RAW_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "route"]
cat_levels = {}


def _num(s: pd.Series) -> pd.Series:
    """c-7 -> 7 (Month/DayofMonth/DayOfWeek arrive as c-<n> strings)."""
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def _fit_levels():
    route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
    cat_levels["UniqueCarrier"] = pd.Index(sorted(train["UniqueCarrier"].dropna().unique()))
    cat_levels["Origin"] = pd.Index(sorted(train["Origin"].dropna().unique()))
    cat_levels["Dest"] = pd.Index(sorted(train["Dest"].dropna().unique()))
    cat_levels["route"] = pd.Index(sorted(route.dropna().unique()))


_fit_levels()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    df = df.copy()
    mon = _num(df["Month"])
    dom = _num(df["DayofMonth"])
    dow = _num(df["DayOfWeek"])
    dep = df["DepTime"].astype(int)
    hour = (dep // 100).clip(0, 24)
    minute = dep - hour * 100
    tod = hour + minute / 60.0  # time of day in hours
    doy = (mon - 1) * 30.5 + dom  # rough position in year

    X = pd.DataFrame(index=df.index)
    X["DepTime"] = dep
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["Month"] = mon
    X["DayofMonth"] = dom
    X["DayOfWeek"] = dow
    X["mon_sin"] = np.sin(2 * np.pi * mon / 12.0)
    X["mon_cos"] = np.cos(2 * np.pi * mon / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    X["Distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    X["route"] = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).values
    X["UniqueCarrier"] = df["UniqueCarrier"].values
    X["Origin"] = df["Origin"].values
    X["Dest"] = df["Dest"].values
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X_all = prepare(train)
y_all = to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=SEED, stratify=y_all)

model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=5,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
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
