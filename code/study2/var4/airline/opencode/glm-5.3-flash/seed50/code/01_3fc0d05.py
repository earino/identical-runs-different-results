"""XGBoost binary classifier on airline delays. Only file the agent edits.

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

CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique()))
              for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]}
HOURS = pd.Index([f"h{h:02d}" for h in range(24)])
DIST_BINS = [0, 100, 200, 300, 400, 500, 750, 1000, 1500, 2500, 6000]


def _dep_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dt // 100).where(dt < 2400)
    minute = dt.where(dt < 2400) % 100
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    out = pd.DataFrame({
        "DepHour": ("h" + hour.astype("Int64").astype(str)).fillna("na"),
        "DepMinute": minute,
        "DepTimeSin": np.sin(2 * np.pi * (hour * 60 + minute) / 1440.0),
        "DepTimeCos": np.cos(2 * np.pi * (hour * 60 + minute) / 1440.0),
        "DistanceLog": np.log1p(dist),
        "DistBin": pd.Categorical(pd.cut(dist, DIST_BINS).astype(str)),
    })
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]].copy()
    for c in X.columns:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    derived = _dep_time_feats(df)
    derived["DepHour"] = pd.Categorical(derived["DepHour"], categories=HOURS)
    out = pd.concat([X, derived], axis=1)
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


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
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
