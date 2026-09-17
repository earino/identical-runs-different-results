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
entity_freq = {c: train[c].value_counts() for c in ["UniqueCarrier", "Origin", "Dest"]}


def _num(col):
    # 'c-7' -> 7
    return col.astype("string").str.replace("^c-", "", regex=True).astype(float)


def _inter_key(df, oc):
    return (np.floor(df["DepTime"].astype(float) / 100).clip(0, 24).astype(int).astype(str)
            + "_" + df[oc].astype(str))


_prep_cache = {}


def prepare(df: pd.DataFrame, variant: str = "full") -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    key = (id(df), variant, len(df))
    if key in _prep_cache:
        return _prep_cache[key]
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
    elif variant == "cyctime":
        X["dep_sin"] = np.sin(2 * np.pi * X["minute_of_day"] / 1440)
        X["dep_cos"] = np.cos(2 * np.pi * X["minute_of_day"] / 1440)
    elif variant == "sizefreq":
        X["dep_sin"] = np.sin(2 * np.pi * X["minute_of_day"] / 1440)
        X["dep_cos"] = np.cos(2 * np.pi * X["minute_of_day"] / 1440)
        for c in ["UniqueCarrier", "Origin", "Dest"]:
            X[f"{c}_freq_log"] = np.log1p(df[c].map(entity_freq[c]).fillna(0).astype(float))
    elif variant == "sfr":
        X["dep_sin"] = np.sin(2 * np.pi * X["minute_of_day"] / 1440)
        X["dep_cos"] = np.cos(2 * np.pi * X["minute_of_day"] / 1440)
        for c in ["UniqueCarrier", "Origin", "Dest"]:
            X[f"{c}_freq_log"] = np.log1p(df[c].map(entity_freq[c]).fillna(0).astype(float))
        route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
        X["route_freq_log"] = np.log1p(route.map(route_counts).fillna(0).astype(float))
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    if len(_prep_cache) > 8:
        _prep_cache.clear()
    _prep_cache[key] = X
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: 20-member feature/css-diverse seed ensemble -----------------------
MEMBER_SPECS = [("full", 4, 0.25), ("cyctime", 4, 0.35), ("cyctime", 4, 0.40),
                ("sizefreq", 4, 0.35), ("sfr", 4, 0.35)]
PARAMS = dict(
    n_estimators=100,
    max_depth=30,
    learning_rate=0.05,
    gamma=1,
    reg_lambda=2,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

t0 = time.time()
members = []
for variant, n, css in MEMBER_SPECS:
    Xtr = prepare(train, variant)
    ytr = to_y(train)
    for seed in range(n):
        m = xgb.XGBClassifier(**PARAMS, colsample_bytree=css, random_state=seed)
        m.fit(Xtr, ytr)
        members.append((variant, m))
    del Xtr
print(f"Training time: {time.time() - t0:.1f}s ({len(members)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(prepare(df, v))[:, 1] for v, m in members], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
