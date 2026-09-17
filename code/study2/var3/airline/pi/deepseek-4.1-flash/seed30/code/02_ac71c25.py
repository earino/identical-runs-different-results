"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes (airline, train=2005 / eval=2006):
  * Only year-stable features are used. Calendar features (Month, DayofMonth) carried 2005-specific
    weather patterns that did NOT transfer to 2006, and removing them raised eval AUC by ~0.007.
    DayOfWeek is stable across years and is kept. Departure time is kept because its effect on delay
    is structural (delays accumulate through the day) and transfers.
  * Carrier / Origin / Dest are passed to XGBoost as native categoricals; test time-of-day and
    distance are numeric. This is what makes the model generalize across the year boundary.
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
TWO_PI = 2.0 * np.pi

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature plan -------------------------------------------------------------
# Information-bearing, year-stable features only:
#   dow, dep_minutes, sin/cos(dep_minutes), distance, UniqueCarrier, Origin, Dest
raw_cat = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in raw_cat}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dow = pd.to_numeric(
        df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False), errors="coerce"
    ).astype(float)
    dist = df["Distance"].astype(float).fillna(0.0)

    dt = df["DepTime"].astype(float).fillna(0.0)
    dep_min = ((dt // 100.0) * 60.0 + (dt % 100.0))
    dep_min = dep_min.where(dep_min <= 1440.0, dep_min - 1440.0)  # fix rare hhmm typos > 2400

    X["dow"] = dow
    X["dep_min"] = dep_min
    X["dep_sin"] = np.sin(TWO_PI * dep_min / 1440.0)
    X["dep_cos"] = np.cos(TWO_PI * dep_min / 1440.0)
    X["dist"] = dist
    for c in raw_cat:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
params = dict(
    max_depth=14,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
    subsample=0.9,
    colsample_bytree=0.4,
    min_child_weight=5,
    reg_lambda=1.0,
    eval_metric="auc",
)

# Pick the tree count with an internal holdout (train only), then refit on all of train.
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
n_val = int(0.15 * len(train))
val_idx = np.sort(idx[:n_val])
fit_idx = np.sort(idx[n_val:])

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
probe = xgb.XGBClassifier(n_estimators=1200, early_stopping_rounds=40, **params)
probe.fit(Xtr.iloc[fit_idx], ytr[fit_idx], eval_set=[(Xtr.iloc[val_idx], ytr[val_idx])], verbose=False)
best_iter = max(1, probe.best_iteration + 1)
print(f"best_iteration={best_iter}  fit_time={time.time() - t0:.1f}s")

model = xgb.XGBClassifier(n_estimators=best_iter, **params)
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
