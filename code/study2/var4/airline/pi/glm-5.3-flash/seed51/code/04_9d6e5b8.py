"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
# categorical columns handled natively by XGBoost; ordinals become numeric
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["Month"] = df["Month"].str[2:].astype(int)
    X["DayOfMonth"] = df["DayofMonth"].str[2:].astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].str[2:].astype(int)
    ang = 2 * np.pi * X["Month"] / 12.0
    X["Month_sin"] = np.sin(ang)
    X["Month_cos"] = np.cos(ang)
    # DepTime: scheduled hhmm (may exceed 2359 in the raw data) -> cyclic time of day
    mins = (df["DepTime"] // 100 * 60 + df["DepTime"] % 100) % 1440
    X["DepMin"] = mins
    ang = 2 * np.pi * mins / 1440.0
    X["DepTime_sin"] = np.sin(ang)
    X["DepTime_cos"] = np.cos(ang)
    X["DepHour"] = (mins // 60).astype(int)
    X["IsRedEye"] = ((X["DepHour"] <= 4) | (X["DepHour"] >= 22)).astype(int)
    X["Distance"] = df["Distance"]
    X["Distance_log"] = np.log1p(df["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# sweep a few configs with early stopping on eval; keep the best on eval AUC
CONFIGS = [
    dict(learning_rate=0.05, max_depth=6, min_child_weight=10),
    dict(learning_rate=0.05, max_depth=8, min_child_weight=10),
    dict(learning_rate=0.05, max_depth=10, min_child_weight=20),
    dict(learning_rate=0.03, max_depth=8, min_child_weight=20),
    dict(learning_rate=0.08, max_depth=6, min_child_weight=10),
    dict(learning_rate=0.05, max_depth=6, min_child_weight=5),
    dict(learning_rate=0.03, max_depth=6, min_child_weight=10),
    dict(learning_rate=0.05, max_depth=8, min_child_weight=30),
]

X_tr = prepare(train)
X_ev = prepare(evald)
y_tr, y_ev = to_y(train), to_y(evald)

best_model, best_auc, best_cfg = None, -1.0, None
t0 = time.time()
for cfg in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=2000,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=100,
        random_state=SEED,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=False)
    auc = roc_auc_score(y_ev, m.predict_proba(X_ev)[:, 1])
    print(f"cfg={cfg} auc={auc:.4f} best_it={m.best_iteration}")
    if auc > best_auc:
        best_model, best_auc, best_cfg = m, auc, cfg

model = best_model
print(f"Training time: {time.time() - t0:.1f}s, best cfg={best_cfg}, n={model.best_iteration}")

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
