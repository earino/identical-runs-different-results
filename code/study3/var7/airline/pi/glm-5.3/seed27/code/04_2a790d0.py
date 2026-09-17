"""XGBoost binary classifier for airline departure delay.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature engineering: smoothed target encodings (fit on TRAIN labels only, unseen -> prior) plus raw
numeric/cyclical features. Early stopping picks the tree count on eval.csv (2006, same period as the
hidden holdout); the metric is AUC.
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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
YTRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(YTRAIN.mean())


# --- derived columns (no fitted statistics) -------------------------------------
def derived(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    d["hour"] = dep // 100
    d["minute"] = (dep // 100) * 60 + dep % 100
    hh = np.where(d["hour"] <= 3, d["hour"] + 24, d["hour"])  # day starts at the 4am trough
    d["min_since_4am"] = (hh - 4) * 60 + dep % 100
    d["min30"] = d["minute"] // 30
    d["month_n"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    d["day_n"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    d["dow_n"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    d["dist_log"] = np.log1p(df["Distance"].astype(float))
    d["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    d["org_hour"] = df["Origin"].astype(str) + "_" + d["hour"].astype(str)
    d["car_hour"] = df["UniqueCarrier"].astype(str) + "_" + d["hour"].astype(str)
    return d


# --- target encodings: (name, source column, smoothing m) -----------------------
TEDEF = [
    ("te_hour", "hour", 30),
    ("te_route", "route", 50),
    ("te_Origin", "Origin", 30),
    ("te_Dest", "Dest", 30),
    ("te_UniqueCarrier", "UniqueCarrier", 30),
    ("te_Month", "Month", 30),
    ("te_DayOfWeek", "DayOfWeek", 30),
    ("te_org_hour", "org_hour", 100),
    ("te_car_hour", "car_hour", 100),
    ("te_min30", "min30", 30),
]
TE_SRC = {"Origin": "Origin", "Dest": "Dest", "UniqueCarrier": "UniqueCarrier", "Month": "Month",
          "DayOfWeek": "DayOfWeek"}


def _te_map(col, m):
    src = TE_SRC.get(col, None)
    keys = (train[src].astype(str) if src else derived(train)[col].astype(str))
    g = pd.DataFrame({"k": keys, "y": YTRAIN}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * PRIOR) / (g["count"] + m)


TE_MAPS = {name: _te_map(col, m) for name, col, m in TEDEF}  # full-train maps, used at predict time
DERIVED_TRAIN = derived(train)  # cache; train is never passed to predict_proba


def _keys(df, d, col):
    src = TE_SRC.get(col, None)
    return (df[src].astype(str) if src else d[col].astype(str))


# Out-of-fold TE for the training matrix: a row's own label never leaks into its own encoding.
# predict_proba always goes through prepare(), which uses the full-train TE_MAPS.
def _oof_matrix():
    X = pd.DataFrame(index=train.index)
    kf = KFold(5, shuffle=True, random_state=0)
    for name, col, m in TEDEF:
        keys = _keys(train, DERIVED_TRAIN, col)
        oof = np.empty(len(keys))
        for tri, tei in kf.split(keys):
            g = pd.DataFrame({"k": keys.iloc[tri], "y": YTRAIN[tri]}).groupby("k")["y"].agg(["sum", "count"])
            sm = (g["sum"] + m * PRIOR) / (g["count"] + m)
            oof[tei] = keys.iloc[tei].map(sm).fillna(PRIOR).to_numpy()
        X[name] = oof
    for c in ["min_since_4am", "minute", "hour", "dist_log", "month_n", "day_n", "dow_n"]:
        X[c] = DERIVED_TRAIN[c].to_numpy()
    return X


XTRAIN = _oof_matrix()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = derived(df)
    X = pd.DataFrame(index=df.index)
    for name, col, _ in TEDEF:
        X[name] = _keys(df, d, col).map(TE_MAPS[name]).astype(float).fillna(PRIOR).to_numpy()
    for c in ["min_since_4am", "minute", "hour", "dist_log", "month_n", "day_n", "dow_n"]:
        X[c] = d[c].to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    early_stopping_rounds=100,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(XTRAIN, YTRAIN, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
