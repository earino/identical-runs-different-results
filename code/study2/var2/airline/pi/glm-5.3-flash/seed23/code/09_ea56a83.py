"""Airline delay XGBoost — experiment 2: parse strings to ints, hour/minute features,
native categoricals for carrier/origin/dest, deeper model with early stopping.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# smoothed target encoding of carrier x hour (fit on train only)
TE_CH_SMOOTH = 50.0
_y = (train[TARGET] == POSITIVE).astype(float)
_ch_key = train["UniqueCarrier"].str.cat((train["DepTime"] // 100).clip(0, 23).astype(str), sep="|")
_ch_stats = pd.DataFrame({"k": _ch_key, "y": _y}).groupby("k").y.agg(["sum", "count"])
TE_CH_MAP = ((_ch_stats["sum"] + TE_CH_SMOOTH * _y.mean()) / (_ch_stats["count"] + TE_CH_SMOOTH)).to_dict()
_CH_PRIOR = float(_y.mean())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    X["month"] = df["Month"].str[2:].astype(int)
    X["dom"] = df["DayofMonth"].str[2:].astype(int)
    X["dow"] = df["DayOfWeek"].str[2:].astype(int)
    dt = df["DepTime"]
    X["deptime"] = dt
    X["hour"] = (dt // 100).clip(0, 23)
    X["minute"] = dt % 100
    X["dt5"] = dt // 5
    X["te_carrier_hour"] = df["UniqueCarrier"].str.cat((dt // 100).clip(0, 23).astype(str), sep="|").map(TE_CH_MAP).fillna(_CH_PRIOR).astype(float)
    dist = df["Distance"].astype(float)
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
    eval_metric="auc",
)
N_SEEDS = 2

# (max_depth, subsample, n_estimators); -1 marks lossguide with max_leaves
CONFIGS = [(6, 0.9, 800), (8, 0.9, 800), (10, 0.8, 800), (-1, 0.9, 800), (12, 0.7, 600)]

t0 = time.time()
X = prepare(train)
y = to_y(train)
models = []
for d, ss, nt in CONFIGS:
    extra = {} if d > 0 else {"grow_policy": "lossguide", "max_leaves": 64}
    for i in range(N_SEEDS):
        m = xgb.XGBClassifier(n_estimators=nt, max_depth=max(d, 0), subsample=ss, random_state=SEED + i, **{k: v for k, v in PARAMS.items() if k not in ("max_depth", "subsample")}, **extra)
        m.fit(X, y)
        models.append(m)
_P = [m.predict_proba(prepare(evald))[:, 1] for m in models]
for i, p in enumerate(_P):
    print(f"member {i}: eval_auc={roc_auc_score(to_y(evald), p):.4f}")
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    P = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return P


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), np.mean(_P, axis=0))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
