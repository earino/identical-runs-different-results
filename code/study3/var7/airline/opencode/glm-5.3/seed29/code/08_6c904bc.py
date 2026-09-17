"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering --------------------------------------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
C_PARSE = ["Month", "DayofMonth", "DayOfWeek"]
# levels fit on the TRAINING data only; unseen values in new data -> NaN category
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
route_levels = pd.Index(sorted((train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).unique()))


# --- smoothed target encoding (fit on TRAIN only; train rows get out-of-fold values) ---
TE_COLS = ["route", "Origin", "Dest", "UniqueCarrier", "origin_hour", "dest_hour", "carrier_hour", "route_hour"]
TE_M = 20.0
y_all = (train[TARGET] == POSITIVE).astype(float)
global_mean = float(y_all.mean())


def te_key(df: pd.DataFrame, name: str) -> pd.Series:
    hour = df["DepTime"] // 100
    if name == "route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    if name == "origin_hour":
        return df["Origin"].astype(str) + "_" + hour.astype(str)
    if name == "dest_hour":
        return df["Dest"].astype(str) + "_" + hour.astype(str)
    if name == "carrier_hour":
        return df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)
    if name == "route_hour":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str) + "_" + hour.astype(str)
    return df[name].astype(str)


def fit_te(keys: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["mean", "count"])
    gm = float(np.mean(y))
    return ((g["mean"] * g["count"] + gm * m) / (g["count"] + m)).astype(float)


te_maps = {c: fit_te(te_key(train, c), y_all.to_numpy(), TE_M) for c in TE_COLS}

# out-of-fold TE for the training rows themselves (avoids leakage into the fit)
oof_te = {c: np.empty(len(train)) for c in TE_COLS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
idx = np.arange(len(train))
for tr_idx, va_idx in kf.split(idx):
    y_tr_f = y_all.to_numpy()[tr_idx]
    gm_f = float(np.mean(y_tr_f))
    for c in TE_COLS:
        k_full = te_key(train, c)
        te = fit_te(pd.Series(k_full.to_numpy()[tr_idx]), y_tr_f, TE_M)
        oof_te[c][va_idx] = pd.Series(k_full.to_numpy()[va_idx]).map(te).fillna(gm_f).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in C_PARSE:
        X[c] = df[c].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["hour"] = df["DepTime"] // 100
    X["minute"] = df["DepTime"] % 100
    X["dep_minutes"] = X["hour"] * 60 + X["minute"]
    X["Distance"] = df["Distance"]
    X["log_distance"] = np.log1p(df["Distance"].astype(float))
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route"] = pd.Categorical(route, categories=route_levels)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    for c in TE_COLS:
        X["te_" + c] = te_key(df, c).map(te_maps[c]).astype(float).fillna(global_mean)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2500,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=80,
    subsample=0.5,
    colsample_bytree=0.5,
    reg_lambda=20.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    random_state=SEED,
    n_jobs=N_JOBS,
    callbacks=[xgb.callback.EarlyStopping(rounds=100, maximize=True)],
)

t0 = time.time()
X_tr = prepare(train)
for c in TE_COLS:  # training rows use out-of-fold TE values (no leakage)
    X_tr["te_" + c] = oof_te[c]
model.fit(
    X_tr,
    to_y(train),
    eval_set=[(prepare(evald), to_y(evald))],
    verbose=50,
)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
