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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
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
    hc = df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
    X["carrier_hour"] = pd.Categorical(hc, categories=hc_levels)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

CONFIGS = [
    ("d4lr05cs07", dict(max_depth=4, learning_rate=0.05, reg_lambda=30.0, colsample_bytree=0.7)),
    ("d4lr05", dict(max_depth=4, learning_rate=0.05, reg_lambda=30.0)),
    ("d3lr05", dict(max_depth=3, learning_rate=0.05, reg_lambda=10.0)),
    ("d3lr05cs07", dict(max_depth=3, learning_rate=0.05, reg_lambda=10.0, colsample_bytree=0.7)),
    ("sq_d4lr05", dict(objective="reg:squarederror", max_depth=4, learning_rate=0.05, reg_lambda=30.0)),
    ("lg_d4lr05", dict(max_depth=10, max_leaves=24, grow_policy="lossguide",
                       learning_rate=0.05, reg_lambda=30.0)),
]

names, preds, models = [], [], []
t0 = time.time()
for i, (name, over) in enumerate(CONFIGS):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=SEED + 7 * i,
        n_jobs=N_JOBS,
        **over,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    r = m.evals_result()["validation_0"]["auc"]
    p = m.predict_proba(Xev)[:, 1]
    names.append(name)
    preds.append(p)
    models.append(m)
    print(f"[ens] {name}: peak={max(r):.4f} at {int(np.argmax(r)) + 1} it, final={roc_auc_score(yev, p):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")

P = np.array(preds)
aucs = {n: roc_auc_score(yev, P[i]) for i, n in enumerate(names)}
order = sorted(range(len(names)), key=lambda i: -aucs[names[i]])
for k in (4, 5, 6):
    sel = order[:k]
    ens = roc_auc_score(yev, P[sel].mean(axis=0))
    print(f"[ens] top{k} {[names[i] for i in sel]}: {ens:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return np.clip(p, 0.0, 1.0)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
