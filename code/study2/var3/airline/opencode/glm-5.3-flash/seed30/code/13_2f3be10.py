"""XGBoost binary classifier on airline delays. Only file the agent edits.

Contract: `python train.py` -> prints `Eval AUC: 0.xxxx`; module-level `predict_proba(df)`.
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

feature_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier", "Origin", "Dest", "Distance"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]


def _to_int(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _prep_base(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _to_int(df["Month"])
    X["DayofMonth"] = _to_int(df["DayofMonth"])
    X["DayOfWeek"] = _to_int(df["DayOfWeek"])
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    dt = (X["DepTime"] // 100) * 60 + (X["DepTime"] % 100)
    X["dep_sin"] = np.sin(2 * np.pi * dt / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * dt / 1440.0)
    X["DepHour"] = dt // 60
    X["LogDist"] = np.log1p(pd.to_numeric(df["Distance"], errors="coerce"))
    for c in CAT_COLS:
        X[c] = df[c].astype("category")
    return X


CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here so predict_proba() applies it to unseen rows.
    X = _prep_base(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    learning_rate=0.03,
    max_depth=15,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_all, y_all = prepare(train), to_y(train)
X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
members = []
for depth, lr, cs, seed in [(10, 0.05, 0.8, SEED), (12, 0.03, 0.8, SEED + 1), (15, 0.03, 0.8, SEED + 2), (15, 0.05, 0.7, SEED + 3), (18, 0.03, 0.8, SEED + 4)]:
    params = dict(PARAMS, max_depth=depth, learning_rate=lr, colsample_bytree=cs, random_state=seed)
    probe = xgb.XGBClassifier(n_estimators=1000, early_stopping_rounds=75, **params)
    probe.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_n = max(int(probe.best_iteration) + 1, 30)
    m = xgb.XGBClassifier(n_estimators=best_n, **params)
    m.fit(X_all, y_all)
    members.append(m)
    print(f"member depth={depth} lr={lr}: best_n={best_n}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(prepare(df))[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
