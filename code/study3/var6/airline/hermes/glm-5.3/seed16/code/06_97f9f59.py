"""Airline delay XGBoost — two-view bagged ensemble (30-min and 10-min time-TE views).

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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- encodings (fit on TRAIN only) ----------------------------------------------
y_tr = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(y_tr.mean())


def smoothed_te(by: pd.Series, m: float) -> pd.Series:
    stats = y_tr.groupby(by).agg(["mean", "count"])
    return (stats["mean"] * stats["count"] + PRIOR * m) / (stats["count"] + m)


TE_ORIGIN = smoothed_te(train["Origin"], 20)
TE_H30 = smoothed_te((train["DepTime"] // 100) * 2, 20)
TE_H10 = smoothed_te((train["DepTime"] // 100) * 6 + (train["DepTime"] % 100) // 10, 20)
CNT_O = train["Origin"].value_counts()
CNT_D = train["Dest"].value_counts()
CNT_OH = (train["Origin"].astype(str) + "_" + (train["DepTime"] // 100).astype(str)).value_counts()

# the two time-TE views: view name -> (TE series, key function on raw DepTime)
VIEWS = {
    "h30": (TE_H30, lambda t: (t // 100) * 2),
    "h10": (TE_H10, lambda t: (t // 100) * 6 + (t % 100) // 10),
}


def prepare(df: pd.DataFrame, view: str = "h30") -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    t = df["DepTime"].astype(int)
    mins = (t % 2400 // 100) * 60 + (t % 100)  # minutes since midnight
    X["mins"] = mins
    X["te_O"] = df["Origin"].map(TE_ORIGIN).astype(float).values
    X["hour_x_dist"] = (t // 100) * df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].astype(float))
    X["sin_m"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_m"] = np.cos(2 * np.pi * mins / 1440.0)
    te, keyfn = VIEWS[view]
    X["te_time"] = keyfn(t).map(te).astype(float).values
    X["cnt_o"] = df["Origin"].map(CNT_O).astype(float).values
    X["cnt_d"] = df["Dest"].map(CNT_D).astype(float).values
    X["cnt_oh"] = (df["Origin"].astype(str) + "_" + (t // 100).astype(str)).map(CNT_OH).astype(float).values
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- two-view bagged ensemble ---------------------------------------------------
N_PER_VIEW = 8
BASE = dict(
    n_estimators=200,
    max_depth=4,
    min_child_weight=50,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
models = []  # (view, fitted model)
for view in ("h30", "h10"):
    X_view = prepare(train, view)
    for i in range(N_PER_VIEW):
        m = xgb.XGBClassifier(random_state=100 + i, **BASE)
        m.fit(X_view, to_y(train))
        models.append((view, m))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    # group models by view so each is scored on its own feature view
    ps = []
    for view, m in models:
        ps.append(m.predict_proba(prepare(df, view))[:, 1])
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
