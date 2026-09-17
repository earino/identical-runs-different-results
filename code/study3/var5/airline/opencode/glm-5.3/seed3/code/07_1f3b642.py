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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
cat_cols = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
INTER_COLS = [("hour_dow", "DayOfWeek"), ("hour_carrier", "UniqueCarrier"), ("hour_origin", "Origin")]
inter_levels = {
    name: pd.Index(sorted(
        (np.floor(train["DepTime"].astype(float) / 100).clip(0, 24).astype(int).astype(str)
         + "_" + train[oc].astype(str)).unique()
    ))
    for name, oc in INTER_COLS
}
route_counts = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()


def _num(col):
    # 'c-7' -> 7
    return col.astype("string").str.replace("^c-", "", regex=True).astype(float)


def _inter_key(df, oc):
    return (np.floor(df["DepTime"].astype(float) / 100).clip(0, 24).astype(int).astype(str)
            + "_" + df[oc].astype(str))


def prepare(df: pd.DataFrame, variant: str = "full") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month"] = _num(df["Month"])
    X["day"] = _num(df["DayofMonth"])
    X["dow"] = _num(df["DayOfWeek"])
    dt = df["DepTime"].astype(float)
    X["dep_time"] = dt
    X["hour"] = np.floor(dt / 100).clip(0, 24)
    X["minute_of_day"] = (np.floor(dt / 100) * 60 + dt % 100).clip(0, 24 * 60)
    X["dist"] = df["Distance"].astype(float)
    X["dist_log"] = np.log1p(X["dist"])
    if variant == "full":
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        X["doy"] = (X["month"] - 1) * 30.44 + X["day"]
        for name, oc in INTER_COLS:
            X[name] = pd.Categorical(_inter_key(df, oc), categories=inter_levels[name])
        route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
        X["route_freq_log"] = np.log1p(route.map(route_counts).fillna(0).astype(float))
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: 8-member feature-diverse seed ensemble ---------------------------
PARAMS = dict(
    n_estimators=150,
    max_depth=20,
    learning_rate=0.05,
    gamma=2,
    reg_lambda=5,
    colsample_bytree=0.3,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
members = []
for seed in range(5):
    m = xgb.XGBClassifier(**PARAMS, random_state=seed)
    m.fit(prepare(train, "full"), to_y(train))
    members.append(("full", m))
for seed in range(3):
    m = xgb.XGBClassifier(**PARAMS, random_state=seed)
    m.fit(prepare(train, "basic"), to_y(train))
    members.append(("basic", m))
print(f"Training time: {time.time() - t0:.1f}s ({len(members)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(prepare(df, v))[:, 1] for v, m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
