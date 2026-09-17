"""Experiment 2: time-feature engineering + ordinal/one-hot encoding + adaptive tree budget.

Changes vs baseline:
- Month/DayofMonth/DayOfWeek parsed to ordinal ints; dayofyear added (seasonality/trend).
- DepTime -> dep_min (minutes since midnight, 2400+ wraps), dep_missing flag.
- UniqueCarrier one-hot; Origin/Dest stay native categorical (282 levels).
- Tree count calibrated in-script from a 30-tree probe so the fit lands ~75s wall.
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

ORDINALS = ["Month", "DayofMonth", "DayOfWeek"]
ONEHOT = ["UniqueCarrier"]
NAT_CATS = ["Origin", "Dest"]
carrier_levels = sorted(train["UniqueCarrier"].astype(str).unique())
nat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in NAT_CATS}
CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)


def _ord(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here; everything below only uses constants fit on train.
    X = pd.DataFrame(index=df.index)
    for c in ORDINALS:
        X[c] = _ord(df[c])
    month = X["Month"].fillna(1).clip(1, 12).astype(int)
    X["dayofyear"] = CUM_DAYS[month - 1] + X["DayofMonth"].fillna(15)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_missing"] = dt.isna().astype(float)
    dtf = dt.fillna(0)
    h = (dtf // 100).astype(float) % 24
    m = dtf % 100
    X["dep_min"] = h * 60 + m
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").fillna(0)
    carr = pd.Categorical(df["UniqueCarrier"].astype(str), categories=carrier_levels)
    dummies = pd.get_dummies(carr, prefix="car").astype(float)
    dummies.index = X.index
    X = pd.concat([X, dummies], axis=1)
    for c in NAT_CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=nat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


FIT_SECONDS = float(os.environ.get("FIT_SECONDS", "75"))
N_MAX = 3000


def calibrated_n_estimators(X, y) -> int:
    probe = xgb.XGBClassifier(
        n_estimators=30, max_depth=6, learning_rate=0.1, tree_method="hist",
        enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    )
    t0 = time.time()
    probe.fit(X, y)
    per_tree = (time.time() - t0) / 30.0
    n = int(min(N_MAX, max(30, FIT_SECONDS / max(per_tree, 1e-6))))
    print(f"probe: {per_tree*1000:.0f} ms/tree -> {n} trees")
    return n


t0 = time.time()
rng = np.random.RandomState(SEED)
idx = rng.permutation(len(train))
val_idx, tr_idx = idx[: len(train) // 5], idx[len(train) // 5 :]
Xtr, ytr = prepare(train.iloc[tr_idx]), to_y(train.iloc[tr_idx])
Xval, yval = prepare(train.iloc[val_idx]), to_y(train.iloc[val_idx])
n_est = calibrated_n_estimators(Xtr, ytr)
model = xgb.XGBClassifier(
    n_estimators=n_est, max_depth=6, learning_rate=0.1, tree_method="hist",
    enable_categorical=True, random_state=SEED, n_jobs=N_JOBS,
    eval_metric="auc",
)
model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
val_auc = roc_auc_score(yval, model.predict_proba(Xval)[:, 1])
print(f"fit {n_est} trees in {time.time() - t0:.1f}s; val AUC (80% train): {val_auc:.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
