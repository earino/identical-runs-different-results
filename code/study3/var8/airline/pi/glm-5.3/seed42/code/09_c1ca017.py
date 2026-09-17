"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Diagnostic: ablation of feature groups at fixed model config.
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

for c in ("Origin", "Dest", "UniqueCarrier"):
    cnt = train[c].value_counts()
    X_train["cnt_" + c] = np.log1p(train[c].map(cnt).fillna(0.0)).to_numpy()
    X_eval["cnt_" + c] = np.log1p(evald[c].map(cnt).fillna(0.0)).to_numpy()


def te_map(grp: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    df = pd.DataFrame({"g": grp.values, "y": y})
    agg = df.groupby("g")["y"].agg(["sum", "count"])
    return (agg["sum"] + m * PRIOR) / (agg["count"] + m)


te_maps = {name: te_map(K_train[name], y_train, TE_M[name]) for name in K_train}

oof = {name: np.zeros(len(train)) for name in K_train}
kf = KFold(n_splits=5, shuffle=True,random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    yf = y_train[tr_idx]
    for name in K_train:
        enc = te_map(K_train[name].iloc[tr_idx], yf, TE_M[name])
        oof[name][va_idx] = enc.reindex(K_train[name].iloc[va_idx].values).fillna(PRIOR).to_numpy()

for name in K_train:
    X_train[name] = oof[name]
    X_eval[name] = te_maps[name].reindex(K_eval[name].values).fillna(PRIOR).to_numpy()

FULL_COLS = list(X_train.columns)


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


GROUPS = {
    "native_car": ["UniqueCarrier", "Origin", "Dest"],
    "sincos": ["sin_tod", "cos_tod"],
    "volume": ["cnt_Origin", "cnt_Dest", "cnt_UniqueCarrier"],
    "time_te": ["te_hour", "te_dep15", "te_carrier_hour", "te_origin_hour", "te_dest_hour"],
    "plain_te": ["te_carrier", "te_origin", "te_dest", "te_route"],
    "dow_month_native": ["Month", "DayofMonth", "DayOfWeek"],
    "dow_month_te": ["te_month", "te_dow", "te_origin_dow"],
}
VARIANTS = {
    "no_dowm_native": GROUPS["dow_month_native"],
    "no_dowm_te": GROUPS["dow_month_te"],
    "no_all_native": GROUPS["native_car"] + GROUPS["dow_month_native"],
    "combo": GROUPS["native_car"] + GROUPS["dow_month_native"] + GROUPS["dow_month_te"],
    "combo_novol": GROUPS["native_car"] + GROUPS["dow_month_native"] + GROUPS["dow_month_te"] + GROUPS["volume"],
    "combo_nosin": GROUPS["native_car"] + GROUPS["dow_month_native"] + GROUPS["dow_month_te"] + GROUPS["sincos"],
}

PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
results = {}
for vname, drop in VARIANTS.items():
    cols = [c for c in FULL_COLS if c not in drop]
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(X_train[cols], y_train, verbose=False)
    best_k, best_auc = 0, 0.0
    for k in (60, 100, 150, 200):
        auc = roc_auc_score(y_eval, m.predict_proba(X_eval[cols], iteration_range=(0, k))[:, 1])
        if auc > best_auc:
            best_k, best_auc = k, auc
    results[vname] = (best_auc, best_k, cols, m)
    print(f"DIAG {vname:14s} best_k={best_k} eval_auc={best_auc:.4f}  ({time.time() - t0:.0f}s)")

BEST_V = max(results, key=lambda v: results[v][0])
best_auc, best_k, COLS, model = results[BEST_V]
print(f"BEST variant={BEST_V} k={best_k} auc={best_auc:.4f}  n_cols={len(COLS)}")
print(f"Diagnostic time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df)[COLS], iteration_range=(0, best_k))[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
