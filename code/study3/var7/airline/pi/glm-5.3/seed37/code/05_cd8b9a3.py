"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Notes from experiments so far:
  - 2005(train) -> 2006(eval/holdout) drift: heavy capacity (deep trees, many rounds at high lr,
    route-level features, target encodings) all HURT eval AUC. Sweet spot: depth 4, lr 0.02, ~450 rounds.
  - DepTime > 2400 means post-midnight (hh up to 26); wrapping hour to %24 helps.
  - Monotone constraint on dep_h (delay prob rises through the day) transfers well.
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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
# feature order matters for monotone_constraints: [DepTime, Distance, hour, minute, dep_h, cats...]
FEATURES = ["DepTime", "Distance", "hour", "minute", "dep_h"] + CAT_COLS
MONO = (0, 0, 0, 0, 1)  # dep_h monotone increasing


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    hour = (df["DepTime"] // 100).astype(float) % 24.0  # DepTime can be 24xx/25xx = post-midnight
    minute = (df["DepTime"] % 100).astype(float)
    X["hour"] = hour
    X["minute"] = minute
    X["dep_h"] = hour + minute / 60.0
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[FEATURES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=700,
    max_depth=14,
    learning_rate=0.02,
    min_child_weight=1,
    reg_lambda=1.0,
    reg_alpha=3.0,
    colsample_bytree=0.4,
    max_bin=512,
    monotone_constraints=MONO,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
