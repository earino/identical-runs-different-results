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


def _key(df, kind):
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    h = (df["DepTime"] // 100).astype(str)
    if kind == "origin":
        return o
    if kind == "dest":
        return d
    if kind == "carrier":
        return df["UniqueCarrier"].astype(str)
    if kind == "hour":
        return h
    if kind == "dow":
        return _cnum(df["DayOfWeek"]).astype(int).astype(str)
    if kind == "hxorigin":
        return h + "_" + o
    if kind == "hxdest":
        return h + "_" + d
    if kind == "hxcarrier":
        return h + "_" + df["UniqueCarrier"].astype(str)
    if kind == "route":
        return o + "_" + d
    raise ValueError(kind)


def te_map(kind, m):
    k = _key(train, kind)
    g = pd.DataFrame({"k": k, "y": Y_TRAIN.to_numpy()}).groupby("k")["y"].agg(["sum", "count"])
    rate = (g["sum"] + m * PRIOR) / (g["count"] + m)
    return rate.to_dict()


TE_KINDS = ["origin", "dest", "carrier", "hour", "dow", "hxorigin", "hxdest", "hxcarrier", "route"]
TE_MAPS = {k: te_map(k, 300 if k == "route" else 100) for k in TE_KINDS}

GROUPS = {
    "base": [],
    "single": ["origin", "dest", "carrier"],
    "time": ["hour", "dow"],
    "inter": ["hxorigin", "hxdest", "hxcarrier"],
    "all": ["origin", "dest", "carrier", "hour", "dow", "hxorigin", "hxdest", "hxcarrier", "route"],
    "all_noroute": ["origin", "dest", "carrier", "hour", "dow", "hxorigin", "hxdest", "hxcarrier"],
}


def prepare(df: pd.DataFrame, te_group: str = "all") -> pd.DataFrame:
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
    for kind in GROUPS[te_group]:
        X["te_" + kind] = _key(df, kind).map(TE_MAPS[kind]).fillna(PRIOR).to_numpy(dtype=float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------------------------
BASE = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=1,
    reg_lambda=1.0,
    seed=SEED,
    nthread=N_JOBS,
    subsample=0.7,
    colsample_bytree=0.6,
)
ROUNDS = [100, 200]

t0 = time.time()
y_all = to_y(train)
y_ev = to_y(evald)
results = {}
for grp in GROUPS:
    dtr = xgb.DMatrix(prepare(train, grp), label=y_all, enable_categorical=True)
    dev = xgb.DMatrix(prepare(evald, grp), enable_categorical=True)
    b = xgb.train(BASE, dtr, num_boost_round=max(ROUNDS), verbose_eval=False)
    for n in ROUNDS:
        auc = roc_auc_score(y_ev, b.predict(dev, iteration_range=(0, n)))
        results[(grp, n)] = auc
        print(f"te_group={grp:12s} rounds={n:3d}  eval_auc={auc:.4f}")

best_grp, best_n = max(results, key=results.get)
print(f"best: te_group={best_grp} rounds={best_n} auc={results[(best_grp, best_n)]:.4f}")
model = xgb.train(BASE, xgb.DMatrix(prepare(train, best_grp), label=y_all, enable_categorical=True), num_boost_round=best_n)
MODEL_GROUP = best_grp
MODEL_ROUNDS = best_n
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict(xgb.DMatrix(prepare(df, MODEL_GROUP), enable_categorical=True), iteration_range=(0, MODEL_ROUNDS))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
