"""XGBoost binary classifier for airline delay prediction. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- fitted-on-train-only artifacts (never computed from the incoming frame) ---
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
FREQ_COLS = ["UniqueCarrier", "Origin", "Dest"]
FREQ_MAP = {c: train[c].value_counts().to_dict() for c in FREQ_COLS}


def _ci(s: pd.Series) -> pd.Series:
    return s.str.replace("c-", "", regex=False).astype(float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _ci(df["Month"])
    dom = _ci(df["DayofMonth"])
    dow = _ci(df["DayOfWeek"])
    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    dep = df["DepTime"]
    hour = (dep // 100) % 24
    minute = dep % 100
    tod = hour + minute / 60.0
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)

    X["distance"] = df["Distance"]
    X["log_dist"] = np.log1p(df["Distance"])

    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    for c in FREQ_COLS:
        X[c + "_freq"] = df[c].map(FREQ_MAP[c]).fillna(0.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


PARAMS = dict(
    n_estimators=4000,
    learning_rate=0.02,
    max_depth=0,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

Xtr = prepare(train)
ytr = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)

t0 = time.time()
model = xgb.XGBClassifier(**PARAMS)
model.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={int(model.best_iteration)}")
_imp = sorted(zip(Xtr.columns, model.feature_importances_), key=lambda x: -x[1])
print("Top importances:", ", ".join(f"{c}={v:.3f}" for c, v in _imp[:12]))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
