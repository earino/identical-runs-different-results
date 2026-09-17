"""XGBoost airline delay classifier: 5-fold bagged ensemble, regularized.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
All statistics (categorical levels) are fit on data/train.csv only.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(pd.unique(train[c])) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here; levels above were fit on train only.
    X = pd.DataFrame(index=df.index)
    X["month"] = pd.to_numeric(df["Month"].astype(str).str.slice(2), errors="coerce")
    X["day"] = pd.to_numeric(df["DayofMonth"].astype(str).str.slice(2), errors="coerce")
    X["dow"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.slice(2), errors="coerce")
    X["deptime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_EST = 200
N_FOLDS = 10


def make_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=N_EST,
        learning_rate=0.05,
        max_depth=24,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.6,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
X_eval = prepare(evald)
y_eval = to_y(evald)

models = []
oof = np.zeros(len(train))
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for k, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
    m = make_model(SEED + k)
    m.fit(X_all.iloc[tr_idx], y_all[tr_idx])
    oof[va_idx] = m.predict_proba(X_all.iloc[va_idx])[:, 1]
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")
print(f"OOF AUC: {roc_auc_score(y_all, oof):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
