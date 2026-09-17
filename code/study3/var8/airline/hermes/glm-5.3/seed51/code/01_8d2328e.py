"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
RAW_NUM = ["DepTime", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
C_NUM = {"Month": 12, "DayofMonth": 31, "DayOfWeek": 7}


def fe(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls fe() on unseen rows,
    # so anything computed on `train`/`evald` outside this function will NOT be applied
    # to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c, n in C_NUM.items():
        v = df[c].str.slice(2).astype(int)  # 'c-7' -> 7
        X[c + "_i"] = v
        X[f"{c}_sin"] = np.sin(2 * np.pi * v / n)
        X[f"{c}_cos"] = np.cos(2 * np.pi * v / n)
    hour = (df["DepTime"] // 100).clip(0, 24).astype(int)
    minute = (df["DepTime"] % 100).astype(int)
    mins = hour * 60 + minute
    X["hour"] = hour
    X["dep_minutes"] = mins
    X["dep_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["distance_log"] = np.log1p(df["Distance"].astype(float))
    X["distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = df[c]
    return X


cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = fe(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


feature_cols = list(fe(train).columns)


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
# internal split for early stopping (train is 2005, eval is 2006; keep time split honest)
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
cut = int(0.9 * len(train))
tr_i, va_i = idx[:cut], idx[cut:]
Xp = prepare(train)
y = to_y(train)
m_es = xgb.XGBClassifier(early_stopping_rounds=50, **PARAMS)
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
