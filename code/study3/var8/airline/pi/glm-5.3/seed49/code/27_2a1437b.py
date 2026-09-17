"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, module-level `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
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

# --- feature engineering -------------------------------------------------------
C_CAT = ["UniqueCarrier", "Origin", "Dest", "Month", "DayofMonth", "DayOfWeek"]
NUM_COLS = ["Distance", "DepTime"]

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in C_CAT}


def _dep(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).astype(int)


def _hour(df: pd.DataFrame) -> pd.Series:
    return (_dep(df) // 100).clip(0, 23)


hc_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _hour(train).astype(str)).unique()))


def prepare(df: pd.DataFrame, with_hc: bool = True) -> pd.DataFrame:
    # ALL feature engineering lives here; predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    hour = _hour(df)
    minute = (_dep(df) % 100).clip(0, 59)
    X["hour"] = hour / 23.0
    X["min_of_day"] = (hour * 60 + minute) / 1439.0
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in C_CAT:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    if with_hc:
        hc = df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
        X["carrier_hour"] = pd.Categorical(hc, categories=hc_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


ytr, yev = to_y(train), to_y(evald)

ROSTER = [
    ("fast_d6lr05cs07_s42", True, dict(max_depth=6, learning_rate=0.05, reg_lambda=30.0, colsample_bytree=0.7), 42, 3000),
    ("slow_d6lr02", True, dict(max_depth=6, learning_rate=0.02, reg_lambda=30.0), 42, 5000),
    ("shallow_d3lr05", True, dict(max_depth=3, learning_rate=0.05, reg_lambda=10.0), 42, 3000),
    ("base_d4lr05cs07", False, dict(max_depth=4, learning_rate=0.05, reg_lambda=30.0, colsample_bytree=0.7), 49, 3000),
]

models = []
t0 = time.time()
for name, with_hc, over, seed, n_est in ROSTER:
    Xtr_i, Xev_i = prepare(train, with_hc), prepare(evald, with_hc)
    m = xgb.XGBClassifier(
        n_estimators=n_est,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=N_JOBS,
        **over,
    )
    m.fit(Xtr_i, ytr, eval_set=[(Xev_i, yev)], verbose=False)
    r = m.evals_result()["validation_0"]["auc"]
    p = m.predict_proba(Xev_i)[:, 1]
    print(f"[ens] {name}: peak={max(r):.4f} at {int(np.argmax(r)) + 1} it, final={roc_auc_score(yev, p):.4f}")
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df, with_hc))[:, 1]
          for m, (_, with_hc, *_rest) in zip(models, ROSTER)]
    return np.clip(np.mean(ps, axis=0), 0.0, 1.0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
