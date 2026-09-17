"""XGBoost binary classifier with engineered features. Only file the agent edits.

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {}
HM30_LEVELS = pd.Index([])
HM15_LEVELS = pd.Index([])

ENSEMBLE = [  # (max_depth, seed): shallow trees + low lr resist the 2005->2006 shift
    (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1),
]
PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    reg_alpha=1.0,
    subsample=0.9,
    colsample_bynode=0.5,
    tree_method="hist",
    enable_categorical=True,
)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering applied identically everywhere (train/eval/holdout)."""
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].astype(float) % 2400
    hour = (dt // 100).astype(int)
    minute = (dt % 100).astype(int)
    hm = (hour * 60 + minute).astype(int)
    X["deptime"] = dt
    X["hour"] = hour
    X["minute"] = minute
    ang = 2 * np.pi * (dt / 1440.0)
    X["distance"] = df["Distance"].astype(float)
    X["logdist"] = np.log1p(X["distance"])
    for c in CAT_COLS:
        X[c] = df[c]
    X["carrier_hm30"] = df["UniqueCarrier"].astype(str) + "_" + (hm // 30).astype(str)
    X["hm15cat"] = (hm // 15).astype(int).astype(str)
    return X


def fit_state() -> None:
    """Fit encoders on TRAINING data only."""
    global HM30_LEVELS, HM15_LEVELS
    for c in CAT_COLS:
        CAT_LEVELS[c] = pd.Index(sorted(train[c].dropna().unique()))
    dt = train["DepTime"].astype(float) % 2400
    hm = ((dt // 100).astype(int)) * 60 + (dt % 100).astype(int)
    HM30_LEVELS = pd.Index(sorted(
        (train["UniqueCarrier"].astype(str) + "_" + (hm // 30).astype(int).astype(str)).unique()))
    HM15_LEVELS = pd.Index(sorted((hm // 15).astype(int).astype(str).unique()))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows."""
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])
    X["carrier_hm30"] = pd.Categorical(X["carrier_hm30"], categories=HM30_LEVELS)
    X["hm15cat"] = pd.Categorical(X["hm15cat"], categories=HM15_LEVELS)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


fit_state()

# --- model --------------------------------------------------------------------
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
models = []
for depth, seed in ENSEMBLE:
    m = xgb.XGBClassifier(max_depth=depth, random_state=seed, n_jobs=N_JOBS, **PARAMS)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
