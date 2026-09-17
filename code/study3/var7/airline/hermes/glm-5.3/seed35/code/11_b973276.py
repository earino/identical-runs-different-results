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
SEED = 7

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature spec (all levels/stats fit on train only) ----------------------------
CAT_RAW = ["UniqueCarrier", "Origin", "Dest"]
CAL = ["Month", "DayofMonth", "DayOfWeek"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_RAW}
_dh = ((pd.to_numeric(train["DepTime"], errors="coerce") // 100) % 24).astype(int)
hc_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _dh.astype(str)).unique()))
oh_counts = (train["Origin"].astype(str) + "_" + _dh.astype(str)).value_counts()
dh_counts = (train["Dest"].astype(str) + "_" + _dh.astype(str)).value_counts()
route_counts = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # calendar "c-4" -> 4, as integers (monotone time features generalize across years)
    for c in CAL:
        X[c] = pd.to_numeric(df[c].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    dh = ((dep // 100) % 24).astype(int)
    X["DepTime"] = dep
    X["DepHour"] = dh
    X["DepMinute"] = dep % 100
    X["MinSinceMid"] = dh * 60 + (dep % 100)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    car = df["UniqueCarrier"].astype(str)
    X["HourCarrier"] = pd.Categorical(car + "_" + dh.astype(str), categories=hc_levels)
    X["n_OriginHour"] = np.log1p((org + "_" + dh.astype(str)).map(oh_counts).fillna(0).to_numpy())
    X["n_DestHour"] = np.log1p((dst + "_" + dh.astype(str)).map(dh_counts).fillna(0).to_numpy())
    X["n_Route"] = np.log1p((org + "_" + dst).map(route_counts).fillna(0).to_numpy())
    X["DistPerMin"] = X["Distance"] / (X["MinSinceMid"] / 60.0 + 1)
    for c in CAT_RAW:
        X[c] = pd.Categorical(car if c == "UniqueCarrier" else (org if c == "Origin" else dst),
                              categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble of XGBoost models (colsample diversity + seed variation) ----------
# recency weighting: 2005 is one year before eval/holdout 2006; later months matter more
month_num = pd.to_numeric(train["Month"].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce").to_numpy()
REC_w = (0.5 + 0.5 * ((month_num - 1) / 11.0) ** 2).astype("float32")

N_MODELS = 7
members = []
t0 = time.time()
for s in range(1, N_MODELS + 1):
    m = xgb.XGBClassifier(
        n_estimators=4000,
        max_depth=18,
        learning_rate=0.05,
        reg_lambda=0.5,
        colsample_bytree=0.6,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=SEED + s,
        n_jobs=N_JOBS,
    )
    m.fit(prepare(train), to_y(train), sample_weight=REC_w, eval_set=[(prepare(evald), to_y(evald))], verbose=False)
    members.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in members]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
