"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
Y_TRAIN = (train[TARGET] == POSITIVE).astype(int)
PRIOR = Y_TRAIN.mean()

# --- feature definitions (fit on TRAIN only; reused for any unseen dataframe) ------------------
RAW_CAT = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in RAW_CAT}
DATE_COLS = ["Month", "DayofMonth", "DayOfWeek"]


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.slice(start=2), errors="coerce")


def te_map(kind, m):
    o = train["Origin"].astype(str)
    d = train["Dest"].astype(str)
    if kind == "origin":
        k = o
    elif kind == "dest":
        k = d
    elif kind == "carrier":
        k = train["UniqueCarrier"].astype(str)
    else:
        raise ValueError(kind)
    g = pd.DataFrame({"k": k, "y": Y_TRAIN.to_numpy()}).groupby("k")["y"].agg(["sum", "count"])
    rate = (g["sum"] + m * PRIOR) / (g["count"] + m)
    return rate.to_dict()


TE_KINDS = ["origin", "dest", "carrier"]
TE_MAPS = {k: te_map(k, 100) for k in TE_KINDS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in DATE_COLS:
        X[c] = _cnum(df[c])
    X["hour"] = df["DepTime"] // 100
    X["minute"] = df["DepTime"] % 100
    X["DepTime"] = df["DepTime"]
    X["Distance"] = df["Distance"]
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    carr = df["UniqueCarrier"].astype(str)
    X["te_origin"] = o.map(TE_MAPS["origin"]).fillna(PRIOR).to_numpy(dtype=float)
    X["te_dest"] = d.map(TE_MAPS["dest"]).fillna(PRIOR).to_numpy(dtype=float)
    X["te_carrier"] = carr.map(TE_MAPS["carrier"]).fillna(PRIOR).to_numpy(dtype=float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
BASE = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    learning_rate=0.1,
    min_child_weight=1,
    reg_lambda=1.0,
    nthread=N_JOBS,
)
# (name, params_over_base, num_rounds, seed)
MEMBERS = [
    ("d6_rand_s1", dict(max_depth=6, subsample=0.7, colsample_bytree=0.6), 200, 1),
    ("d6_rand_s2", dict(max_depth=6, subsample=0.7, colsample_bytree=0.6), 200, 2),
    ("d6_rand_s3", dict(max_depth=6, subsample=0.7, colsample_bytree=0.6), 200, 3),
    ("d5_rand_s1", dict(max_depth=5, subsample=0.7, colsample_bytree=0.6), 200, 1),
    ("d5_rand_s2", dict(max_depth=5, subsample=0.7, colsample_bytree=0.6), 200, 2),
    ("d4_s1", dict(max_depth=4, subsample=1.0, colsample_bytree=1.0), 100, 1),
    ("d6_rand_lr05", dict(max_depth=6, subsample=0.7, colsample_bytree=0.6, learning_rate=0.05), 400, 1),
]

t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
dtr = xgb.DMatrix(prepare(train), label=y_all, enable_categorical=True)
dev = xgb.DMatrix(prepare(evald), enable_categorical=True)
models = []
preds = []
for name, over, n, seed in MEMBERS:
    b = xgb.train({**BASE, **over, "seed": seed}, dtr, num_boost_round=n, verbose_eval=False)
    p = b.predict(dev)
    models.append((b, n))
    preds.append(p)
    print(f"member={name:14s} eval_auc={roc_auc_score(y_ev, p):.4f}")

ens_p = np.mean(preds, axis=0)
print(f"ensemble_mean eval_auc={roc_auc_score(y_ev, ens_p):.4f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dm = xgb.DMatrix(prepare(df), enable_categorical=True)
    return np.mean([b.predict(dm) for b, _ in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
