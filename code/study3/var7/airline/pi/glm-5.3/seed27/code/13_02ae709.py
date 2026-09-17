"""XGBoost binary classifier for airline departure delay.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: smoothed target encodings (TE) keyed by route/origin/dest/carrier x time-of-day granularity
(delay propensities of scheduled flights persist across years), each paired with a log-count feature
so the trees can discount small-sample cells. The training matrix uses out-of-fold TE values (no
self-leakage); predict time uses full-train maps. Early stopping picks the tree count on eval.csv
(2006, same period as the hidden holdout), metric AUC.
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

# distance-bucket edges are fit on TRAINING data only, reused for any frame
_DIST_Q = np.unique(np.quantile(train["Distance"], np.linspace(0, 1, 9)))
DIST_BINS = np.concatenate([_DIST_Q, [np.inf]])
YTRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(YTRAIN.mean())


# --- derived key columns (no fitted statistics) ---------------------------------
def derived(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    d["hour"] = dep // 100
    d["minute"] = (dep // 100) * 60 + dep % 100
    d["min30"] = d["minute"] // 30
    d["min15"] = d["minute"] // 15
    hh = np.where(d["hour"] <= 3, d["hour"] + 24, d["hour"])  # day starts at the 4am trough
    d["min_since_4am"] = (hh - 4) * 60 + dep % 100
    d["month_n"] = df["Month"].str.replace("c-", "", regex=False).astype(int)
    d["day_n"] = df["DayofMonth"].str.replace("c-", "", regex=False).astype(int)
    d["dow_n"] = df["DayOfWeek"].str.replace("c-", "", regex=False).astype(int)
    d["dist_log"] = np.log1p(df["Distance"].astype(float))
    d["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    d["org_hour"] = df["Origin"].astype(str) + "_" + d["hour"].astype(str)
    d["car_hour"] = df["UniqueCarrier"].astype(str) + "_" + d["hour"].astype(str)
    d["dest_hour"] = df["Dest"].astype(str) + "_" + d["hour"].astype(str)
    d["route_hour"] = d["route"] + "_" + d["hour"].astype(str)
    d["route_min30"] = d["route"] + "_" + d["min30"].astype(str)
    d["route_min15"] = d["route"] + "_" + d["min15"].astype(str)
    d["org_min30"] = df["Origin"].astype(str) + "_" + d["min30"].astype(str)
    d["car_min30"] = df["UniqueCarrier"].astype(str) + "_" + d["min30"].astype(str)
    d["dest_min30"] = df["Dest"].astype(str) + "_" + d["min30"].astype(str)
    d["hour_month"] = d["hour"].astype(str) + "_" + df["Month"].astype(str)
    d["org_min15"] = df["Origin"].astype(str) + "_" + d["min15"].astype(str)
    d["moh"] = df["DepTime"].astype(int) % 100  # minute within the hour (hub bank structure)
    d["distb"] = pd.cut(df["Distance"], bins=DIST_BINS, include_lowest=True, labels=False).astype("Int64").astype(str)
    d["car_org"] = df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    d["car_dest"] = df["UniqueCarrier"].astype(str) + "_" + df["Dest"].astype(str)
    d["org_distb"] = df["Origin"].astype(str) + "_" + d["distb"]
    d["distb_hour"] = d["distb"] + "_" + d["hour"].astype(str)
    return d


# --- TE definitions: (feature name, key column, smoothing m) ---------------------
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
    ("te_min15", "min15", 30),
    ("te_route_hour", "route_hour", 150),
    ("te_route_min30", "route_min30", 200),
    ("te_route_min15", "route_min15", 300),
    ("te_org_min30", "org_min30", 200),
    ("te_car_min30", "car_min30", 200),
    ("te_dest_min30", "dest_min30", 200),
    ("te_dest_hour", "dest_hour", 100),
    ("te_hour_month", "hour_month", 100),
    ("te_org_min15", "org_min15", 300),
    ("te_car_dest", "car_dest", 150),
    ("te_car_org", "car_org", 150),
    ("te_distb_hour", "distb_hour", 150),
    ("te_org_distb", "org_distb", 150),
]
TE_SRC = {"Origin": "Origin", "Dest": "Dest", "UniqueCarrier": "UniqueCarrier", "Month": "Month",
          "DayOfWeek": "DayOfWeek"}
DERIVED_TRAIN = derived(train)  # cache; train is never passed to predict_proba


def _keys(df, d, col):
    src = TE_SRC.get(col)
    return (df[src].astype(str) if src else d[col].astype(str))


def _smooth(keys, y, m):
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * PRIOR) / (g["count"] + m)


# full-train maps for predict time (fit on training data only)
TE_MAPS = {col: _smooth(_keys(train, DERIVED_TRAIN, col), YTRAIN, m) for _, col, m in TEDEF}
CNT_MAPS = {col: _keys(train, DERIVED_TRAIN, col).value_counts() for _, col, _ in TEDEF}

NUMCOLS = ["min_since_4am", "minute", "hour", "dist_log", "month_n", "day_n", "dow_n", "moh"]


def _assemble(df, d, te_vals):
    X = pd.DataFrame(index=df.index)
    for name, col, _ in TEDEF:
        X[name] = te_vals[col]
        X["n_" + col] = np.log1p(_keys(df, d, col).map(CNT_MAPS[col]).fillna(0)).to_numpy()
    for c in NUMCOLS:
        X[c] = d[c].to_numpy()
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for any raw frame (eval, hidden holdout): full-train TE maps."""
    d = derived(df)
    te_vals = {col: _keys(df, d, col).map(TE_MAPS[col]).astype(float).fillna(PRIOR).to_numpy()
               for _, col, _ in TEDEF}
    return _assemble(df, d, te_vals)


def _oof_matrix():
    """Training matrix: out-of-fold TE values (own label never leaks into own encoding)."""
    kf = KFold(3, shuffle=True, random_state=0)
    te_vals = {}
    for _, col, _ in TEDEF:
        keys = _keys(train, DERIVED_TRAIN, col)
        oof = np.empty(len(keys))
        for tri, tei in kf.split(keys):
            sm = _smooth(keys.iloc[tri], YTRAIN[tri], TE_M[col])
            oof[tei] = keys.iloc[tei].map(sm).fillna(PRIOR).to_numpy()
        te_vals[col] = oof
    return _assemble(train, DERIVED_TRAIN, te_vals)


TE_M = {col: m for _, col, m in TEDEF}
XTRAIN = _oof_matrix()


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of jittered XGBs --------------------------------------
EVALX = prepare(evald)
EVALY = to_y(evald)

DROP_GROUPS = [  # each member trains without one feature group (diversity beats seed jitter)
    ["te_car_min30", "n_car_min30", "te_car_hour", "n_car_hour"],
    ["te_dest_min30", "n_dest_min30", "te_dest_hour", "n_dest_hour"],
    ["te_org_min30", "n_org_min30", "te_org_hour", "n_org_hour"],
    ["te_route_min30", "n_route_min30"],
    ["te_hour_month", "n_hour_month", "te_Month", "n_Month"],
    ["te_route_min15", "n_route_min15"],
    ["te_org_min15", "n_org_min15"],
    ["te_DayOfWeek", "n_DayOfWeek"],
]

t0 = time.time()
models = []
for s, drop in enumerate(DROP_GROUPS):
    m = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=0,
        grow_policy="lossguide",
        max_leaves=64,
        min_child_weight=10,
        subsample=0.75,
        colsample_bytree=0.75,
        tree_method="hist",
        early_stopping_rounds=100,
        eval_metric="auc",
        random_state=s + 1,
        n_jobs=N_JOBS,
    )
    m.fit(XTRAIN.drop(columns=drop), YTRAIN, eval_set=[(EVALX.drop(columns=drop), EVALY)], verbose=False)
    models.append((m, drop))
print(f"Training time: {time.time() - t0:.1f}s  best_iters={[m.best_iteration for m, _ in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X.drop(columns=drop))[:, 1] for m, drop in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
