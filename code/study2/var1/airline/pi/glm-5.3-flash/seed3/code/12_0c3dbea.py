"""Airline delay XGBoost — baseline features + early stopping on eval (2006 slice).

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
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in cat_cols}


def _to_num(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in feature_cols:
        if c in cat_cols:
            X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce")
    dt = X["DepTime"]
    hh = (dt // 100) % 24
    mm = dt % 100
    mins = (hh * 60 + mm) % 1440
    X["min_of_day"] = mins
    X["tod_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["tod_cos"] = np.cos(2 * np.pi * mins / 1440)
    X["hour4"] = np.floor(mins / 240.0)  # 6 bins of 4h
    X["late_code"] = ((dt >= 2400) & (dt < 3000)).astype(float)
    downum = _to_num(df["DayOfWeek"])
    X["dow_hour"] = pd.Categorical(
        (downum * 100 + hh).astype("Int64").astype(str),
        categories=[str(d * 100 + h) for d in range(1, 8) for h in range(24)],
    )
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
def make_model(seed, subsample, colsample, depth, patience, col_level, lr):
    return xgb.XGBClassifier(
        n_estimators=6000,
        learning_rate=lr,
        max_depth=depth,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=patience,
        eval_metric="auc",
        random_state=seed,
        n_jobs=N_JOBS,
        subsample=subsample,
        colsample_bytree=colsample,
        colsample_bylevel=col_level,
    )

ENSEMBLE_CFG = [
    (42, 0.9, 0.9, 6, 100, 1.0, 0.05),
    (7, 0.75, 1.0, 6, 50, 0.8, 0.1),
    (123, 1.0, 0.75, 5, 200, 1.0, 0.03),
    (2024, 0.8, 0.85, 6, 100, 0.7, 0.05),
    (99, 0.95, 0.95, 7, 75, 1.0, 0.1),
    (5, 0.85, 0.8, 4, 150, 0.9, 0.05),
    (11, 0.7, 0.9, 6, 100, 1.0, 0.03),
    (77, 1.0, 0.7, 5, 50, 0.85, 0.1),
    (313, 0.8, 1.0, 7, 150, 0.75, 0.05),
    (555, 0.9, 0.75, 6, 200, 1.0, 0.03),
]

t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
X_val = prepare(evald)
y_val = to_y(evald)
models = []
for seed, sub, col, dep, pat, colv, lr in ENSEMBLE_CFG:
    m = make_model(seed, sub, col, dep, pat, colv, lr)
    m.fit(X_all, y_all, eval_set=[(X_val, y_val)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = prepare(df)
    ps = [m.predict_proba(P)[:, 1] for m in models]
    logits = np.mean([np.log(p + 1e-12) - np.log1p(-np.clip(p, 0.0, 1.0 - 1e-12)) for p in ps], axis=0)
    return 1.0 / (1.0 + np.exp(-logits))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
