"""XGBoost binary classifier for the airline delay task. THE ONLY FILE THE AGENT EDITS.

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

# --- encoders (fitted on training data only) -----------------------------------
CAT_LEVELS = {
    "Month": [f"c-{i}" for i in range(1, 13)],
    "DayofMonth": [f"c-{i}" for i in range(1, 32)],
    "DayOfWeek": [f"c-{i}" for i in range(1, 8)],
}
for c in ("UniqueCarrier", "Origin", "Dest"):
    CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = dt // 100
    minute = dt % 100

    # scheduled time of day: smooth cyclical + decision-friendly categorical slots
    X["dt_hour"] = hour
    X["dt_min"] = minute
    X["dt_sin"] = np.sin(2 * np.pi * (hour + minute / 60.0) / 24.0)
    X["dt_cos"] = np.cos(2 * np.pi * (hour + minute / 60.0) / 24.0)
    X["dt_slot"] = pd.Categorical(
        hour.clip(0, 26).astype(int).astype(str).radd("h"),
        categories=[f"h{i}" for i in range(27)],
    )

    # calendar
    for c, levels in CAT_LEVELS.items():
        X[c] = pd.Categorical(df[c], categories=levels)

    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["Distance"])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_weight=20,
    reg_lambda=5.0,
    reg_alpha=1.0,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
Xtr = prepare(train)
ytr = to_y(train)
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
