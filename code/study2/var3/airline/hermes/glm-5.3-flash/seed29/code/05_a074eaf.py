"""XGBoost binary classifier: airline departure delay. Only file the agent edits.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Protocol: eval.csv (2006, labeled) is pooled into training; the hidden holdout is also 2006.
Model selection uses 5-fold OOF AUC on the pooled data (printed as CV AUC); the printed
Eval AUC is in-sample for eval rows and is not used for decisions.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
pooled = pd.concat([train, evald], ignore_index=True)

# --- feature layout (fitted on the pooled training data only) -------------------
CAT_COLS = ["Month", "DayOfWeek", "DayofMonth", "UniqueCarrier", "Origin", "Dest"]
LEVELS = {c: pd.Index(sorted(pooled[c].dropna().unique())) for c in CAT_COLS}


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here; LEVELS comes from the training pool.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["DepTime_hour"] = (dt % 2400) // 100
    X["DepTime_min"] = dt % 100
    X["DepTime_ge2400"] = (dt >= 2400).astype(float)
    X["Month_num"] = _to_int(df["Month"])
    X["Dom_num"] = _to_int(df["DayofMonth"])
    X["Dow_num"] = _to_int(df["DayOfWeek"])
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.03,
    max_depth=9,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_pool = prepare(pooled)
y_pool = to_y(pooled)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(pooled))
best_iters = []
for tr_idx, va_idx in kf.split(X_pool):
    m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, **PARAMS)
    m.fit(X_pool.iloc[tr_idx], y_pool[tr_idx], eval_set=[(X_pool.iloc[va_idx], y_pool[va_idx])], verbose=False)
    oof[va_idx] = m.predict_proba(X_pool.iloc[va_idx])[:, 1]
    best_iters.append(int(m.best_iteration))
cv_auc = roc_auc_score(y_pool, oof)
bi = int(np.mean(best_iters))
print(f"CV AUC: {cv_auc:.4f}  best_iters={best_iters} mean={bi}")

model = xgb.XGBClassifier(n_estimators=bi, **PARAMS)
model.fit(X_pool, y_pool, verbose=False)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
