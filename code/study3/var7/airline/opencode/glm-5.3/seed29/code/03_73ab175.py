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
TE_COLS = ["route", "Origin", "Dest", "UniqueCarrier"]
TE_M = 20.0
y_all = (train[TARGET] == POSITIVE).astype(float)
global_mean = float(y_all.mean())
te_maps = {}
for c in TE_COLS:
    key = train["Origin"].astype(str) + "_" + train["Dest"].astype(str) if c == "route" else train[c].astype(str)
    g = key.to_frame("k").assign(y=y_all.to_numpy()).groupby("k")["y"].agg(["mean", "count"])
    te_maps[c] = ((g["mean"] * g["count"] + global_mean * TE_M) / (g["count"] + TE_M)).astype(float)

# out-of-fold TE for the training rows themselves (avoids leakage into the fit)
oof_te = {c: np.empty(len(train)) for c in TE_COLS}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
idx = np.arange(len(train))
for tr_idx, va_idx in kf.split(idx):
    y_tr_f = y_all.to_numpy()[tr_idx]
    gm = float(y_tr_f.mean())
    for c in TE_COLS:
        key = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)) if c == "route" else train[c].astype(str)
        k_va = key.to_numpy()[va_idx]
        k_tr = key.to_numpy()[tr_idx]
        df_g = pd.DataFrame({"k": k_tr, "y": y_tr_f}).groupby("k")["y"].agg(["mean", "count"])
        te = (df_g["mean"] * df_g["count"] + gm * TE_M) / (df_g["count"] + TE_M)
        oof_te[c][va_idx] = pd.Series(k_va).map(te).fillna(gm).to_numpy()


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
        key = route if c == "route" else df[c].astype(str)
        X["te_" + c] = key.map(te_maps[c]).astype(float).fillna(global_mean)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ----------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=2500,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
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
