"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
RAW_NUM_FROM_STR = ["Month", "DayofMonth", "DayOfWeek"]
cat_cols = [c for c in cat_cols if c not in RAW_NUM_FROM_STR]
feature_cols = [c for c in feature_cols if c not in RAW_NUM_FROM_STR]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _cnum(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    X["Month"] = _cnum(df["Month"])
    X["DayofMonth"] = _cnum(df["DayofMonth"])
    X["DayOfWeek"] = _cnum(df["DayOfWeek"])
    dep = _num(df["DepTime"])
    X["DepHour"] = dep // 100
    X["MinuteOfDay"] = dep // 100 * 60 + dep % 100
    X["sin_hour"] = np.sin(2 * np.pi * X["MinuteOfDay"] / 1440)
    X["cos_hour"] = np.cos(2 * np.pi * X["MinuteOfDay"] / 1440)
    # compact numeric interactions
    X["hour_x_dist"] = X["DepHour"] * _num(df["Distance"])
    X["dow_x_hour"] = X["DayOfWeek"] * 100 + X["DepHour"]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- bagged XGB ensemble ------------------------------------------------------
Xtr_full = prepare(train)
ytr = to_y(train)
rng = np.random.default_rng(SEED)
N_MODELS = 15
models = []
t0 = time.time()
for i in range(N_MODELS):
    col_take = rng.choice(len(Xtr_full.columns), size=max(4, int(0.75 * Xtr_full.shape[1])), replace=False)
    cols_i = list(Xtr_full.columns[col_take])
    m = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=24,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        subsample=1.0,
        colsample_bytree=0.6,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr_full[cols_i], ytr)
    models.append((m, cols_i))
print(f"Training time: {time.time() - t0:.1f}s for {N_MODELS} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X[cols])[:, 1] for m, cols in models]
    ps = np.clip(np.asarray(ps), 1e-6, 1 - 1e-6)
    lo = np.mean(np.log(ps / (1 - ps)), axis=0)  # mean log-odds
    return 1 / (1 + np.exp(-lo))  # back to probability


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
