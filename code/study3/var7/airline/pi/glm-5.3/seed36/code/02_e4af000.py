"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _cnum(s: pd.Series) -> pd.Series:
    """c-7 -> 7"""
    return s.str.slice(2).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    mon = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    dep = df["DepTime"].astype(int)
    hour = (dep // 100) % 24
    minute = dep % 100
    X["month"] = mon
    X["day"] = day
    X["dow"] = dow
    X["hour"] = hour
    X["minute"] = minute
    # cyclical encodings
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["min_sin"] = np.sin(2 * np.pi * minute / 60)
    X["min_cos"] = np.cos(2 * np.pi * minute / 60)
    X["dep_angle"] = 2 * np.pi * (hour * 60 + minute) / 1440
    X["mon_sin"] = np.sin(2 * np.pi * mon / 12)
    X["mon_cos"] = np.cos(2 * np.pi * mon / 12)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    X["distance"] = df["Distance"].astype(float)
    X["distance_log"] = np.log1p(X["distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(**kw):
    params = dict(
        n_estimators=1500,
        max_depth=10,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        early_stopping_rounds=None,
        random_state=SEED,
        n_jobs=N_JOBS,
    )
    params.update(kw)
    return xgb.XGBClassifier(**params)


t0 = time.time()
Xtr_full = prepare(train)
ytr_full = to_y(train)
X_fit, X_val, y_fit, y_val = train_test_split(Xtr_full, ytr_full, test_size=0.1, random_state=SEED, stratify=ytr_full)
m0 = make_model(early_stopping_rounds=60)
m0.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
best_it = m0.best_iteration
print(f"Round 1: best_iteration={best_it} time={time.time()-t0:.1f}s")
model = make_model(n_estimators=max((m0.best_iteration or 0) + 1, 50))
model.fit(Xtr_full, ytr_full, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
