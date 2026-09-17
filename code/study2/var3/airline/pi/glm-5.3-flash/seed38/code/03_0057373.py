"""XGBoost binary classifier for airline dep_delayed_15min. The only file the agent edits.

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
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_COLS = ["UniqueCarrier", "Origin", "Dest"]  # target-encoded (smoothed)
TE_M = 100  # TE smoothing strength
RAW_COLS = ["DepTime", "Distance"]
feature_cols = CAT_COLS + RAW_COLS
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

_ytr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
_prior = _ytr.mean()
_g = pd.DataFrame({"k": train["UniqueCarrier"].values, "y": _ytr}).groupby("k")["y"].agg(["sum", "count"])
# smoothed target-encoding maps, fit on TRAIN ONLY (module level), applied inside prepare()
TE_MAPS = {}
for c in TE_COLS:
    g = pd.DataFrame({"k": train[c].values, "y": _ytr}).groupby("k")["y"].agg(["sum", "count"])
    TE_MAPS[c] = (((g["sum"] + TE_M * _prior) / (g["count"] + TE_M)).to_dict(), _prior)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[RAW_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = X["DepTime"].fillna(0).astype(int)
    hour = np.minimum(dt // 100, 24)
    tod = hour * 60 + dt % 100
    X["hour"] = hour
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["night"] = ((hour < 6) | (hour >= 24)).astype(int)
    X["evening"] = ((hour >= 16) & (hour < 21)).astype(int)
    for c in TE_COLS:
        mp, prior = TE_MAPS[c]
        X[c + "_te"] = df[c].map(mp).fillna(prior)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)
model = xgb.XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    colsample_bytree=0.25,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X, y)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
