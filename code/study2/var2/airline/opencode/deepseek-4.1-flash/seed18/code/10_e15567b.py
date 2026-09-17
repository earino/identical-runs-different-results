"""XGBoost binary classifier for airline delay prediction.

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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


y_train = to_y(train)

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

SMOOTH = {
    "carrier": 20.0,
    "origin": 30.0,
    "dest": 30.0,
    "route": 10.0,
    "carrier_route": 10.0,
    "origin_hour": 30.0,
    "carrier_hour": 30.0,
    "hour": 30.0,
    "origin_month": 30.0,
    "carrier_month": 30.0,
    "route_dow": 10.0,
    "dest_hour": 30.0,
}
KEY_NAMES = list(SMOOTH)


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce").astype(int)


def key_series(df: pd.DataFrame) -> dict:
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23).astype(int)
    month = _cnum(df["Month"]).astype(str)
    dow = _cnum(df["DayOfWeek"]).astype(str)
    route = origin + "_" + dest
    return {
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "route": route,
        "carrier_route": carrier + "_" + route,
        "origin_hour": origin + "_" + hour.astype(str),
        "carrier_hour": carrier + "_" + hour.astype(str),
        "hour": hour.astype(str),
        "origin_month": origin + "_" + month,
        "carrier_month": carrier + "_" + month,
        "route_dow": route + "_" + dow,
        "dest_hour": dest + "_" + hour.astype(str),
    }


PRIOR = float(y_train.mean())
TE_MAP = {}
FREQ_MAP = {}
OOF_TE = {}
for name, keys in key_series(train).items():
    k = SMOOTH[name]
    g = pd.DataFrame({"k": keys.values, "y": y_train}).groupby("k")["y"].agg(["sum", "count"])
    TE_MAP[name] = (g["sum"] + PRIOR * k) / (g["count"] + k)
    FREQ_MAP[name] = keys.value_counts()
    oof = np.full(len(train), PRIOR, dtype=float)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf.split(keys):
        m = pd.DataFrame({"k": keys.iloc[tr_idx].values, "y": y_train[tr_idx]}).groupby("k")["y"].agg(["sum", "count"])
        m = (m["sum"] + PRIOR * k) / (m["count"] + k)
        oof[va_idx] = keys.iloc[va_idx].map(m).fillna(PRIOR).to_numpy()
    OOF_TE[name] = oof


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)

    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    minute = dep % 100
    tod = hour * 60 + minute
    X["dep_hour"] = hour
    X["dep_min"] = minute
    X["dep_tod"] = tod
    X["dep_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["dep_cos"] = np.cos(2 * np.pi * tod / 1440.0)

    month = _cnum(df["Month"])
    day = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    X["dayofyear"] = (month - 1) * 30.4 + day
    X["is_weekend"] = (dow >= 6).astype(int)

    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_distance"] = np.log1p(dist)

    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])

    for name, keys in key_series(df).items():
        X[name + "_te"] = keys.map(TE_MAP[name]).fillna(PRIOR).to_numpy()
        X[name + "_freq"] = keys.map(FREQ_MAP[name]).fillna(0.0).to_numpy()

    return X


def fit_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = prepare(df)
    for name in KEY_NAMES:
        X[name + "_te"] = OOF_TE[name]
    return X


N_BAGS = 5
MODELS = []
X_TR = fit_matrix(train)
t0 = time.time()
for i in range(N_BAGS):
    m = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.6,
        min_child_weight=20,
        reg_lambda=3.0,
        reg_alpha=0.5,
        gamma=0.1,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    m.fit(X_TR, y_train)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
