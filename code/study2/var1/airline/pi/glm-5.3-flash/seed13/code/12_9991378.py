"""XGBoost binary classifier for airline delays. THE ONLY FILE THE AGENT EDITS.

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
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
carrhour_levels = pd.Index(
    sorted(
        (
            train["UniqueCarrier"].astype(str)
            + "_"
            + (train["DepTime"].astype(float).fillna(0) // 100).astype(int).astype(str)
        ).unique()
    )
)
route_counts = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()


def _cnum(s: pd.Series) -> pd.Series:
    """'c-<n>' string -> numeric n (also tolerates plain numbers)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["Month"] = _cnum(df["Month"])
    X["DayofMonth"] = _cnum(df["DayofMonth"])
    X["DayOfWeek"] = _cnum(df["DayOfWeek"])
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce").astype(float)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)
    # time-of-day encodings
    t = X["DepTime"].fillna(0)
    hour = t // 100
    minute = t % 100
    X["dep_min"] = hour * 60 + minute
    frac = X["dep_min"] / 1440.0
    X["dep_sin"] = np.sin(2 * np.pi * frac)
    X["dep_cos"] = np.cos(2 * np.pi * frac)
    X["dep_sin2"] = np.sin(4 * np.pi * frac)
    X["dep_cos2"] = np.cos(4 * np.pi * frac)
    X["dep_sin3"] = np.sin(6 * np.pi * frac)
    X["dep_cos3"] = np.cos(6 * np.pi * frac)
    X["dep_hour"] = pd.Categorical(hour, categories=pd.Index(range(30)))
    # carrier x scheduled-hour interaction
    ch = df["UniqueCarrier"].astype(str) + "_" + hour.astype(int).astype(str)
    X["carr_hour"] = pd.Categorical(ch, categories=carrhour_levels)
    # route popularity (log count of Origin_Dest pairs in training data)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_freq"] = np.log1p(route.map(route_counts).fillna(0).astype(float))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# deep trees + strong column subsampling; 2-seed bag reduces seed variance (~±0.002)
MEMBERS = [
    dict(max_depth=16, learning_rate=0.015, subsample=0.9, colsample_bytree=0.4, random_state=42),
    dict(max_depth=16, learning_rate=0.015, subsample=0.9, colsample_bytree=0.4, random_state=1),
    dict(max_depth=16, learning_rate=0.015, subsample=0.9, colsample_bytree=0.4, random_state=2),
    dict(max_depth=16, learning_rate=0.015, subsample=0.9, colsample_bytree=0.4, random_state=3),
]

Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
models = []
t0 = time.time()
for mp in MEMBERS:
    m = xgb.XGBClassifier(
        n_estimators=20000,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        n_jobs=N_JOBS,
        **mp,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"member seed={mp['random_state']}: best_iter={m.best_iteration}", flush=True)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
