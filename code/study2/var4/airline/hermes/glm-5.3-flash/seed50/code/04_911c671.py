"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

v3: base categoricals (train-fitted levels) + interaction categoricals with train-fitted
levels: {Origin, UniqueCarrier, Dest, DayOfWeek} x dep-hour, Origin x Distance-block.
All encodings are levels-only (unseen levels -> NaN), fitted on train.csv exclusively.
Regularized trees (depth 5, mcw 20, lambda 200, colsample_bynode 0.8, 150 trees) — heavy
regularization helps because train is 2005 and eval is 2006 (concept drift).
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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
DIST_BINS = [0, 250, 500, 750, 1000, 1500, 2500, 10000]

# encoders fit on TRAIN ONLY ----------------------------------------------------
LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"] // 100).clip(0, 24).astype(int)


def _dist_block(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["Distance"], bins=DIST_BINS, labels=False).astype(int)


def _inter_levels(cat_tr: pd.Series, num_tr: pd.Series) -> pd.Index:
    return pd.Index(sorted((cat_tr.astype(str) + "_" + num_tr.astype(str)).unique()))


_htr = _hour(train)
_dbtr = _dist_block(train)
INTER_LEVELS = {
    "Orig_hour": _inter_levels(train["Origin"], _htr),
    "Car_hour": _inter_levels(train["UniqueCarrier"], _htr),
    "Dest_hour": _inter_levels(train["Dest"], _htr),
    "DOW_hour": _inter_levels(train["DayOfWeek"], _htr),
    "Orig_dist": _inter_levels(train["Origin"], _dbtr),
    "DB_hour": _inter_levels(_dbtr, _htr),
    "Car_dist": _inter_levels(train["UniqueCarrier"], _dbtr),
    "DOW_dist": _inter_levels(train["DayOfWeek"], _dbtr),
}
_INTER_SOURCES = {
    "Orig_hour": ("Origin", "hour"),
    "Car_hour": ("UniqueCarrier", "hour"),
    "Dest_hour": ("Dest", "hour"),
    "DOW_hour": ("DayOfWeek", "hour"),
    "Orig_dist": ("Origin", "dist_block"),
    "DB_hour": ("dist_block", "hour"),
    "Car_dist": ("UniqueCarrier", "dist_block"),
    "DOW_dist": ("DayOfWeek", "dist_block"),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    # Categoricals use train-fitted levels; unseen levels become NaN automatically.
    X = pd.DataFrame(index=df.index)
    h = _hour(df).astype(str)
    db = _dist_block(df)
    db_s = db.astype(str)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=LEVELS[c])
    for name, (src, kind) in _INTER_SOURCES.items():
        key = (db_s if src == "dist_block" else df[src].astype(str)) + "_" + (h if kind == "hour" else db_s)
        X[name] = pd.Categorical(key, categories=INTER_LEVELS[name])
    X["DepTime"] = df["DepTime"].astype(float)
    X["Distance"] = df["Distance"].astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    min_child_weight=20,
    reg_lambda=200.0,
    colsample_bynode=0.8,
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
