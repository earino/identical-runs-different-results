"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
CARRIER = "UniqueCarrier"
cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
train_y_mean = float((train[TARGET] == POSITIVE).mean())


_y = None
train_y = (train[TARGET] == POSITIVE).astype(int).to_numpy()

# count features fit on the training slice: traffic/busyness proxies that are stable year over year
cnt_maps = {}
cnt_maps["route"] = train.groupby(["Origin", "Dest"]).size().to_dict()
cnt_maps["carrier"] = train.groupby(CARRIER).size().to_dict()
cnt_maps["carrier_route"] = train.groupby([CARRIER, "Origin", "Dest"]).size().to_dict()
cnt_maps["origin_hour"] = train.groupby(["Origin", train["DepTime"] // 100]).size().to_dict()
cnt_maps["dest_hour"] = train.groupby(["Dest", train["DepTime"] // 100]).size().to_dict()
cnt_maps["origin"] = train.groupby("Origin").size().to_dict()
cnt_maps["dest"] = train.groupby("Dest").size().to_dict()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"] + cat_cols].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["DepTime_sin"] = np.sin(2 * np.pi * X["DepTime"] / 2400)
    X["DepTime_cos"] = np.cos(2 * np.pi * X["DepTime"] / 2400)
    # numeric versions of the c-encoded date fields (allow threshold splits)
    X["month_n"] = X["Month"].astype(str).str.slice(2).astype(int)
    X["day_n"] = X["DayofMonth"].astype(str).str.slice(2).astype(int)
    X["dow_n"] = X["DayOfWeek"].astype(str).str.slice(2).astype(int)
    X["doy"] = (X["month_n"] - 1) * 31 + X["day_n"]
    X["doy_sin"] = np.sin(2 * np.pi * X["doy"] / 365)
    X["doy_cos"] = np.cos(2 * np.pi * X["doy"] / 365)
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["is_weekend"] = (X["dow_n"] >= 6).astype(int)
    X["log_distance"] = np.log1p(X["Distance"])
    # count / busyness features (fit on train only)
    X["cnt_route"] = pd.MultiIndex.from_arrays([df["Origin"], df["Dest"]]).map(cnt_maps["route"]).astype(float)
    X["cnt_carrier"] = df[CARRIER].map(cnt_maps["carrier"]).astype(float)
    X["cnt_carrier_route"] = pd.MultiIndex.from_arrays([df[CARRIER], df["Origin"], df["Dest"]]).map(cnt_maps["carrier_route"]).astype(float)
    X["cnt_origin"] = df["Origin"].map(cnt_maps["origin"]).astype(float)
    X["cnt_dest"] = df["Dest"].map(cnt_maps["dest"]).astype(float)
    X["cnt_origin_hour"] = pd.MultiIndex.from_arrays([df["Origin"], df["DepTime"] // 100]).map(cnt_maps["origin_hour"]).astype(float)
    X["cnt_dest_hour"] = pd.MultiIndex.from_arrays([df["Dest"], df["DepTime"] // 100]).map(cnt_maps["dest_hour"]).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 5
models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=8,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    t0 = time.time()
    m.fit(
        prepare(train),
        to_y(train),
        eval_set=[(prepare(evald), to_y(evald))],
        verbose=False,
    )
    models.append(m)
    print(f"model {i}: {time.time() - t0:.1f}s, best iters: {m.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
