"""XGBoost binary classifier for airline delay. Contract per program.md.

1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive). All per-row feature
   engineering lives inside prepare(); any fitted statistics come from train only.
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest", "DayOfWeek"]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic per-row transforms (no fitting)."""
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(np.int32)
    X["DepTime"] = dep
    X["hour"] = dep // 100
    X["minute"] = dep % 100
    tm = dep.mod(2400)
    tmin = (tm // 100) * 60 + tm % 100
    ang = 2 * np.pi * tmin / 1440.0
    X["sin_t"] = np.sin(ang)
    X["cos_t"] = np.cos(ang)
    mo = df["Month"].str[2:].astype(int)
    X["MonthNum"] = mo
    a = 2 * np.pi * (mo - 1) / 12.0
    X["sin_m"] = np.sin(a)
    X["cos_m"] = np.cos(a)
    X["DayOfMonth"] = df["DayofMonth"].str[2:].astype(int)
    dw = df["DayOfWeek"].str[2:].astype(int)
    X["DayOfWeekNum"] = dw
    a = 2 * np.pi * dw / 7.0
    X["sin_d"] = np.sin(a)
    X["cos_d"] = np.cos(a)
    dist = df["Distance"].astype(np.float32)
    X["Distance"] = dist
    X["log_dist"] = np.log1p(dist)
    X["UniqueCarrier"] = df["UniqueCarrier"]
    X["Origin"] = df["Origin"]
    X["Dest"] = df["Dest"]
    X["DayOfWeek"] = df["DayOfWeek"]
    return X


TR_FEAT = base_features(train)
CAT_LEVELS = {c: pd.Index(sorted(TR_FEAT[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X = prepare(train)
y = to_y(train)
Xe = prepare(evald)
ye = to_y(evald)


def make_model(n_est: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_est,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


t0 = time.time()
CONFIGS = [
    (0.4, 16, 300, 42),
    (0.5, 12, 300, 42),
    (0.4, 12, 300, 7),
    (0.5, 16, 300, 7),
    (0.3, 12, 400, 123),
    (0.4, 10, 400, 123),
]
members = []
probs = []
for col, depth, n_est, seed in CONFIGS:
    m = xgb.XGBClassifier(
        n_estimators=n_est,
        learning_rate=0.05,
        max_depth=depth,
        subsample=0.9,
        colsample_bytree=col,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(X, y)
    p = m.predict_proba(Xe)[:, 1]
    print(f"member col={col} depth={depth} n_est={n_est} seed={seed} eval_auc={roc_auc_score(ye, p):.4f}")
    members.append(m)
    probs.append(p)
ens = np.mean(probs, axis=0)
ens_auc = roc_auc_score(ye, ens)
print(f"ensemble eval_auc={ens_auc:.4f}")
print(f"Training time: {time.time() - t0:.1f}s")
model = members


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xdf = prepare(df)
    return np.mean([m.predict_proba(Xdf)[:, 1] for m in model], axis=0)


eval_auc = ens_auc
print(f"Eval AUC: {eval_auc:.4f}")
