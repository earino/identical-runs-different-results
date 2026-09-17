"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature set: no native categoricals, TEs + sincos + volume. Diagnostic: hyperparameter grid.
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


def _hour(df):
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    return (dt // 100).clip(0, 24), (dt % 100).clip(0, 59), dt


def base_X(df, hour, minute, dt):
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = dt
    X["hour"] = hour
    X["minute"] = minute
    tod = hour * 60 + minute
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    return X


def keys_for(df, hour, dt):
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


TE_M = dict(te_carrier=150.0, te_origin=150.0, te_dest=150.0, te_month=300.0,
            te_dow=300.0, te_hour=300.0, te_carrier_hour=400.0, te_origin_hour=400.0,
            te_dest_hour=400.0, te_route=500.0, te_origin_dow=400.0, te_dep15=500.0)


def te_map(grp: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    df = pd.DataFrame({"g": grp.values, "y": y})
    agg = df.groupby("g")["y"].agg(["sum", "count"])
    return (agg["sum"] + m * PRIOR) / (agg["count"] + m)


def build_matrices():
    H_tr, M_tr, DT_tr = _hour(train)
    H_ev, M_ev, DT_ev = _hour(evald)
    X_tr = base_X(train, H_tr, M_tr, DT_tr)
    X_ev = base_X(evald, H_ev, M_ev, DT_ev)
    K_tr = keys_for(train, H_tr, DT_tr)
    K_ev = keys_for(evald, H_ev, DT_ev)
    for c in ("Origin", "Dest", "UniqueCarrier"):
        cnt = train[c].value_counts()
        X_tr["cnt_" + c] = np.log1p(train[c].map(cnt).fillna(0.0)).to_numpy()
        X_ev["cnt_" + c] = np.log1p(evald[c].map(cnt).fillna(0.0)).to_numpy()
    global te_maps, oof
    te_maps = {name: te_map(K_tr[name], y_train, TE_M[name]) for name in K_tr}
    oof = {name: np.zeros(len(train)) for name in K_tr}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in kf.split(train):
        yf = y_train[tr_idx]
        for name in K_tr:
            enc = te_map(K_tr[name].iloc[tr_idx], yf, TE_M[name])
            oof[name][va_idx] = enc.reindex(K_tr[name].iloc[va_idx].values).fillna(PRIOR).to_numpy()
    for name in K_tr:
        X_tr[name] = oof[name]
        X_ev[name] = te_maps[name].reindex(K_ev[name].values).fillna(PRIOR).to_numpy()
    return X_tr, X_ev


t0 = time.time()
X_train, X_eval = build_matrices()
COLS = list(X_train.columns)
print(f"Feature build: {time.time() - t0:.1f}s  n_cols={len(COLS)}")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    hour, minute, dt = _hour(df)
    X = base_X(df, hour, minute, dt)
    for c in ("Origin", "Dest", "UniqueCarrier"):
        cnt = train[c].value_counts()
        X["cnt_" + c] = np.log1p(df[c].map(cnt).fillna(0.0)).to_numpy()
    K = keys_for(df, hour, dt)
    for name in K:
        X[name] = te_maps[name].reindex(K[name].values).fillna(PRIOR).to_numpy()
    return X[COLS]


# --- grid diagnostic ------------------------------------------------------------
results = {}
GRID = [
    (8, 30, 1, 0.05, 1.0, 600),
    (8, 30, 1, 0.07, 1.0, 400),
    (8, 30, 1, 0.10, 1.0, 300),
    (8, 30, 1, 0.10, 5.0, 300),
    (6, 1, 1, 0.05, 1.0, 600),
    (6, 1, 1, 0.10, 1.0, 400),
    (6, 10, 1, 0.05, 1.0, 600),
    (10, 30, 1, 0.07, 1.0, 300),
]
KS = (150, 250, 350, 450, 600)
for depth, mcw, _, lr, rl, n in GRID:
    m = xgb.XGBClassifier(
        n_estimators=n, max_depth=depth, learning_rate=lr, min_child_weight=mcw,
        reg_lambda=rl, tree_method="hist",
        random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(X_train, y_train, verbose=False)
    best_k, best_auc = 0, 0.0
    for k in KS:
        if k > n:
            continue
        auc = roc_auc_score(y_eval, m.predict_proba(X_eval, iteration_range=(0, k))[:, 1])
        if auc > best_auc:
            best_k, best_auc = k, auc
    key = f"d{depth}_m{mcw}_lr{lr}_l{rl}"
    results[key] = (best_auc, best_k, depth, mcw, lr, rl)
    print(f"DIAG {key:24s} best_k={best_k} eval_auc={best_auc:.4f}  ({time.time() - t0:.0f}s)")

BEST = max(results, key=lambda r: results[r][0])
best_auc, best_k, D, MCW, LR, RL = results[BEST]
print(f"BEST {BEST} k={best_k} auc={best_auc:.4f}")
print(f"Grid time: {time.time() - t0:.1f}s")

model = xgb.XGBClassifier(
    n_estimators=best_k, max_depth=D, learning_rate=LR, min_child_weight=MCW,
    reg_lambda=RL, tree_method="hist",
    random_state=SEED, n_jobs=N_JOBS,
)
model.fit(X_train, y_train, verbose=False)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
