"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold, train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
STR_INT_COLS = ["Month", "DayofMonth", "DayOfWeek"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

def _base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric features derived from raw columns (no fit state needed)."""
    X = pd.DataFrame(index=df.index)
    for c in STR_INT_COLS:
        X[c] = df[c].str[2:].astype(int)
    dep = df["DepTime"]
    hour = (dep // 100) % 24
    frac = hour + (dep % 100) / 60.0
    X["DepTime"] = dep
    X["dep_hour"] = hour
    X["dep_min"] = dep % 100
    X["hour_sin"] = np.sin(2 * np.pi * frac / 24.0)
    X["hour_cos"] = np.cos(2 * np.pi * frac / 24.0)
    X["month_sin"] = np.sin(2 * np.pi * X["Month"] / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * X["Month"] / 12.0)
    X["dow"] = X["DayOfWeek"]
    X["dow_sin"] = np.sin(2 * np.pi * X["DayOfWeek"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["DayOfWeek"] / 7.0)
    X["dom"] = X["DayofMonth"]
    X["Distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    return _base_frame(df)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all = prepare(train)
y_all = to_y(train)

SEEDS = [42, 7, 13]

X_fit, X_val, y_fit, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

t0 = time.time()
probe = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=16,
    subsample=0.7,
    colsample_bytree=0.6,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=100,
    random_state=SEEDS[0],
    n_jobs=N_JOBS,
)
probe.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
best_n = probe.best_iteration + 1
print(f"probe: best_iteration={best_n}, fit {time.time() - t0:.1f}s")

models = []
for seed in SEEDS:
    t0 = time.time()
    final = xgb.XGBClassifier(
        n_estimators=best_n,
        learning_rate=0.03,
        max_depth=16,
        subsample=0.7,
        colsample_bytree=0.6,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    final.fit(X_all, y_all)
    models.append(final)
    print(f"seed {seed}: fit {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    preds = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
