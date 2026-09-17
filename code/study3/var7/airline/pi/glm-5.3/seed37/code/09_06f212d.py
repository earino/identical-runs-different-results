"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Experiment notes:
  - 2005(train) -> 2006(eval/holdout) drift: route-level target encodings and high-capacity boosting
    memorize 2005-specific noise. What transfers: main effects + robust interactions
    (carrier x hour, origin x hour, dest x hour, carrier x distance bin).
  - "RF-ization" of boosting helps a lot: very deep trees (30) + strong L1 (alpha=3) + colsample 0.3.
  - DepTime > 2400 = post-midnight; wrap hour to %24.
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

# --- feature engineering -------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

HOUR_B = lambda d: d["DepTime"].div(100).mod(24).astype(int)  # scheduled hour (wrapped past midnight)
DIST_B = lambda d: pd.cut(d["Distance"], bins=[0, 300, 600, 1000, 1500, 2500, 6000], labels=False).astype(int)

CODES = {  # robust interaction codes; levels from the TRAINING data only
    "carr_hour": lambda d: d["UniqueCarrier"].astype(str) + "_" + HOUR_B(d).astype(str),
    "orig_hour": lambda d: d["Origin"].astype(str) + "_" + HOUR_B(d).astype(str),
    "dest_hour": lambda d: d["Dest"].astype(str) + "_" + HOUR_B(d).astype(str),
    "carr_dist": lambda d: d["UniqueCarrier"].astype(str) + "_" + DIST_B(d).astype(str),
}
code_levels = {name: pd.Index(sorted(fn(train).unique())) for name, fn in CODES.items()}

FEATURES = ["DepTime", "Distance", "hour", "minute", "dep_h"] + CAT_COLS + list(CODES.keys())


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
    for name, fn in CODES.items():
        X[name] = pd.Categorical(fn(df), categories=code_levels[name])
    return X[FEATURES]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=700,
    max_depth=30,
    learning_rate=0.02,
    min_child_weight=3,
    reg_lambda=1.0,
    reg_alpha=3.0,
    colsample_bytree=0.3,
    max_bin=512,
    max_cat_to_onehot=100,
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
