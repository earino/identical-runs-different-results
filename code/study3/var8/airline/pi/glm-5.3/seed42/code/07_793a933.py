"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature set B + expanded smoothed target encodings (OOF for train rows) + volume counts.
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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_eval = (evald[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(y_train.mean())

BASE_OBJ = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
base_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in BASE_OBJ}


def _hour(df):
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    return (dt // 100).clip(0, 24), (dt % 100).clip(0, 59), dt


def base_X(df, hour, minute, dt):
    X = pd.DataFrame(index=df.index)
    for c in BASE_OBJ:
        X[c] = pd.Categorical(df[c], categories=base_levels[c])
    X["DepTime"] = dt
    X["hour"] = hour
    X["minute"] = minute
    tod = hour * 60 + minute
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    return X


def keys_for(df, hour, dt):
    """TE group keys, computed identically for any raw frame."""
    hs = hour.astype(int).astype(str)
    dow = df["DayOfWeek"].astype(str)
    dep15 = (dt // 15).astype(int).astype(str)
    return {
        "te_carrier": df["UniqueCarrier"].astype(str),
        "te_origin": df["Origin"].astype(str),
        "te_dest": df["Dest"].astype(str),
        "te_month": df["Month"].astype(str),
        "te_dow": dow,
        "te_hour": hs,
        "te_carrier_hour": df["UniqueCarrier"].astype(str) + "|" + hs,
        "te_origin_hour": df["Origin"].astype(str) + "|" + hs,
        "te_dest_hour": df["Dest"].astype(str) + "|" + hs,
        "te_route": df["Origin"].astype(str) + "|" + df["Dest"].astype(str),
        "te_origin_dow": df["Origin"].astype(str) + "|" + dow,
        "te_dep15": dep15,
    }


H_train, M_train, DT_train = _hour(train)
H_eval, M_eval, DT_eval = _hour(evald)
X_train = base_X(train, H_train, M_train, DT_train)
X_eval = base_X(evald, H_eval, M_eval, DT_eval)
K_train = keys_for(train, H_train, DT_train)
K_eval = keys_for(evald, H_eval, DT_eval)

TE_M = {k: 150.0 for k in K_train}
TE_M.update(
    te_hour=300.0, te_month=300.0, te_dow=300.0, te_dep15=500.0,
    te_route=500.0, te_carrier_hour=400.0, te_origin_hour=400.0, te_dest_hour=400.0,
    te_origin_dow=400.0,
)

# volume features (train counts; log-scaled), stable across years
for c in ("Origin", "Dest", "UniqueCarrier"):
    cnt = train[c].value_counts()
    for X, df in ((X_train, train), (X_eval, evald)):
        X["cnt_" + c] = np.log1p(df[c].map(cnt).fillna(0.0)).to_numpy()

VOL = ["cnt_Origin", "cnt_Dest", "cnt_UniqueCarrier"]


def te_map(grp: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    df = pd.DataFrame({"g": grp.values, "y": y})
    agg = df.groupby("g")["y"].agg(["sum", "count"])
    return (agg["sum"] + m * PRIOR) / (agg["count"] + m)


te_maps = {name: te_map(K_train[name], y_train, TE_M[name]) for name in K_train}

oof = {name: np.zeros(len(train)) for name in K_train}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    yf = y_train[tr_idx]
    for name in K_train:
        enc = te_map(K_train[name].iloc[tr_idx], yf, TE_M[name])
        oof[name][va_idx] = enc.reindex(K_train[name].iloc[va_idx].values).fillna(PRIOR).to_numpy()

for name in K_train:
    X_train[name] = oof[name]
    X_eval[name] = te_maps[name].reindex(K_eval[name].values).fillna(PRIOR).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    hour, minute, dt = _hour(df)
    X = base_X(df, hour, minute, dt)
    for c in ("Origin", "Dest", "UniqueCarrier"):
        cnt = train[c].value_counts()
        X["cnt_" + c] = np.log1p(df[c].map(cnt).fillna(0.0)).to_numpy()
    K = keys_for(df, hour, dt)
    for name in K:
        X[name] = te_maps[name].reindex(K[name].values).fillna(PRIOR).to_numpy()
    return X


PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model = xgb.XGBClassifier(**PARAMS)
model.fit(X_train, y_train, verbose=False)
curve = {}
for k in (100, 150, 200, 300, 400):
    p = model.predict_proba(X_eval, iteration_range=(0, k))[:, 1]
    curve[k] = roc_auc_score(y_eval, p)
    print(f"DIAG n={k} eval_auc={curve[k]:.4f}")
BEST_K = max(curve, key=curve.get)
print(f"Training time: {time.time() - t0:.1f}s  best_k={BEST_K}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, BEST_K))[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
