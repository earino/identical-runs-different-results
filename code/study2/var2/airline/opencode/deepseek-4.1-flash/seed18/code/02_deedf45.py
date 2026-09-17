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
}
KEY_NAMES = list(SMOOTH)


def key_series(df: pd.DataFrame) -> dict:
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23).astype(int).astype(str)
    route = origin + "_" + dest
    return {
        "carrier": carrier,
        "origin": origin,
        "dest": dest,
        "route": route,
        "carrier_route": carrier + "_" + route,
        "origin_hour": origin + "_" + hour,
        "carrier_hour": carrier + "_" + hour,
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


model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(fit_matrix(train), y_train)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
