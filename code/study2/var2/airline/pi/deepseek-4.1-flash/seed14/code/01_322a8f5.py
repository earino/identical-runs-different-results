"""XGBoost binary classifier for flight delay prediction.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["route"] = d["Origin"].astype(str) + "_" + d["Dest"].astype(str)
    return d


train = add_derived(pd.read_csv("data/train.csv"))
evald = add_derived(pd.read_csv("data/eval.csv"))

train["_y"] = (train[TARGET] == POSITIVE).astype(int)
LOW_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in LOW_CATS}
BASE_NUM = ["DepTime", "Distance"]

# --- smoothed target encoding (fit on training data only) ---------------------
TE_COLS = ["Origin", "Dest", "UniqueCarrier", "route"]
PRIOR = float((train[TARGET] == POSITIVE).mean())
SMOOTH = 20.0


def _te_map(frame: pd.DataFrame, col: str) -> pd.Series:
    g = frame.groupby(col, observed=True)["_y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * SMOOTH) / (g["count"] + SMOOTH)


te_maps = {c: _te_map(train, c) for c in TE_COLS}

# out-of-fold encodings for the training rows (avoids target leakage)
oof = np.full((len(train), len(TE_COLS)), PRIOR, dtype=float)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    sub = train.iloc[tr_idx]
    for j, c in enumerate(TE_COLS):
        oof[va_idx, j] = train.iloc[va_idx][c].map(_te_map(sub, c)).to_numpy()
OOF = pd.DataFrame(oof, columns=[c + "_te" for c in TE_COLS], index=train.index)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    d = add_derived(df)
    X = d[LOW_CATS + BASE_NUM].copy()
    for c in LOW_CATS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in TE_COLS:
        X[c + "_te"] = d[c].map(te_maps[c]).astype(float).fillna(PRIOR)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr = prepare(train)
for c in TE_COLS:
    Xtr[c + "_te"] = OOF[c + "_te"].to_numpy()

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xtr, to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
