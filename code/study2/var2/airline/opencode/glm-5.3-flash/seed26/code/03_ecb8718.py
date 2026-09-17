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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
HIGH_CARD = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in HIGH_CARD}


def _cnum(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"c-(\d+)", expand=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    dom = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["dom"] = dom
    X["dom_sin"] = np.sin(2 * np.pi * dom / 31)
    X["dom_cos"] = np.cos(2 * np.pi * dom / 31)
    X["dow"] = dow
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    dep = df["DepTime"].astype(float)
    hh = (dep // 100).clip(0, 47)
    mm = dep % 100
    mins = ((hh * 60 + mm) % 1440)
    X["dep_min"] = mins
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["dep_hour"] = mins // 60
    X["distance"] = df["Distance"].astype(float)
    X["distance_log"] = np.log1p(df["Distance"].astype(float))
    for c in HIGH_CARD:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    learning_rate=0.05,
    max_depth=10,
    subsample=0.75,
    colsample_bytree=0.6,
    min_child_weight=20,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = len(train) // 10
val_idx, fit_idx = idx[:n_val], idx[n_val:]
Xfit, yfit = prepare(train.iloc[fit_idx]), to_y(train.iloc[fit_idx])
Xval, yval = prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx])
probe = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=50, **PARAMS)
probe.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
best_n = max(int(probe.best_iteration) + 1, 50)
print(f"probe time: {time.time() - t0:.1f}s  best_iter={probe.best_iteration}")

# refit on ALL training data at the chosen iteration count (a bit more trees: more data)
model = xgb.XGBClassifier(n_estimators=int(best_n * 1.1), **{k: v for k, v in PARAMS.items() if k != "eval_metric"})
t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s  n_trees={model.n_estimators}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
