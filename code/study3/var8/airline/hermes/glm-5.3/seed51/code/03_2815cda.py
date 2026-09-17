"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Notes:
  - Plain categoricals (carrier/origin/dest) + numeric time features beat target encodings here.
  - All feature engineering lives inside fe(); encoders/stats are fit on train only.
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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def fe(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls fe() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str.slice(2).astype(int)
    X["day"] = df["DayofMonth"].str.slice(2).astype(int)
    X["dow"] = df["DayOfWeek"].str.slice(2).astype(int)
    h = (df["DepTime"] // 100).clip(0, 24)
    mins = h * 60 + df["DepTime"] % 100
    X["dep_raw"] = df["DepTime"].astype(int)
    X["hour"] = h
    X["mins"] = mins
    X["sin"] = np.sin(2 * np.pi * mins / 1440)
    X["cos"] = np.cos(2 * np.pi * mins / 1440)
    X["minofhr"] = df["DepTime"] % 100
    X["dist"] = df["Distance"].astype(float)
    X["logdist"] = np.log1p(df["Distance"].astype(float))
    for c in CAT_COLS:
        X[c] = df[c].to_numpy()
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = fe(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    max_depth=20,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=1,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
# internal split for early stopping (random split of the 2005 training slice)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
cut = int(0.9 * len(train))
tr_i, va_i = idx[:cut], idx[cut:]
Xp = prepare(train)
y = to_y(train)
m_es = xgb.XGBClassifier(n_estimators=4000, early_stopping_rounds=60, **PARAMS)
m_es.fit(Xp.iloc[tr_i], y[tr_i], eval_set=[(Xp.iloc[va_i], y[va_i])], verbose=False)
best_round = m_es.best_iteration + 1
print(f"ES rounds: {best_round}, training time {time.time() - t0:.1f}s")

p2 = dict(PARAMS)
p2["n_estimators"] = best_round
model = xgb.XGBClassifier(**p2)
model.fit(Xp, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
