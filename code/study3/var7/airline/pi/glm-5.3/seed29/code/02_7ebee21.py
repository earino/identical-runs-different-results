"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
    """'c-4' -> 4 (falls back to NaN -> -1 for anything unexpected)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").fillna(-1).astype(int)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = _cnum(df["Month"])
    X["day"] = _cnum(df["DayofMonth"])
    X["dow"] = _cnum(df["DayOfWeek"])
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype(int)
    X["dep_hour"] = dt // 100
    X["dep_min"] = dt % 100
    X["dep_minutes"] = X["dep_hour"] * 60 + X["dep_min"]
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_minutes"] / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_minutes"] / 1440.0)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(-1)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.1,
    max_depth=8,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X = prepare(train)
y = to_y(train)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
es = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=50, **PARAMS)
es.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_iter = es.best_iteration + 1
print(f"Training time (ES split): {time.time() - t0:.1f}s, best_iter={best_iter}, val AUC={es.best_score:.4f}")

t0 = time.time()
model = xgb.XGBClassifier(n_estimators=best_iter, **PARAMS)
model.fit(X, y, verbose=False)
print(f"Training time (full): {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
