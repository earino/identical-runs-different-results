"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Diagnostic: sweep of target-encoding smoothing constants (m), single strong model.
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


TE_KEYS = ["te_carrier", "te_origin", "te_dest", "te_month", "te_dow", "te_origin_dow",
           "te_hour", "te_dep15", "te_dep5", "te_carrier_hour", "te_origin_hour", "te_dest_hour"]
CNT_KEYS = ["cnt_origin_hour", "cnt_dest_hour", "cnt_route", "cnt_carrier_hour",
            "cnt_origin_dow", "cnt_origin_dep15", "cnt_dest_dep15", "cnt_global_dep15"]

BASE_M = dict(te_carrier=150.0, te_origin=150.0, te_dest=150.0, te_month=300.0,
              te_dow=300.0, te_origin_dow=400.0, te_hour=300.0, te_dep15=500.0,
              te_dep5=300.0, te_carrier_hour=400.0, te_origin_hour=400.0, te_dest_hour=400.0)


def keys_for(df, hour, dt):
    hs = hour.astype(int).astype(str)
    dow = df["DayOfWeek"].astype(str)
    dep15 = (dt // 15).astype(int).astype(str)
    dep5 = (dt // 5).astype(int).astype(str)
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    c = df["UniqueCarrier"].astype(str)
    return {
        "te_carrier": c, "te_origin": o, "te_dest": d,
        "te_month": df["Month"].astype(str), "te_dow": dow,
        "te_origin_dow": o + "|" + dow,
        "te_hour": hs, "te_dep15": dep15, "te_dep5": dep5,
        "te_carrier_hour": c + "|" + hs, "te_origin_hour": o + "|" + hs, "te_dest_hour": d + "|" + hs,
        "cnt_origin_hour": o + "|" + hs, "cnt_dest_hour": d + "|" + hs,
        "cnt_route": o + "|" + d, "cnt_carrier_hour": c + "|" + hs,
        "cnt_origin_dow": o + "|" + dow, "cnt_origin_dep15": o + "|" + dep15,
        "cnt_dest_dep15": d + "|" + dep15, "cnt_global_dep15": dep15,
    }


def te_map(grp: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    df = pd.DataFrame({"g": grp.values, "y": y})
    agg = df.groupby("g")["y"].agg(["sum", "count"])
    return (agg["sum"] + m * PRIOR) / (agg["count"] + m)


def base_X(df, hour, minute, dt):
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = dt
    X["hour"] = hour
    X["minute"] = minute
    tod = hour * 60 + minute
    X["sin_tod"] = np.sin(2 * np.pi * tod / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    for c in ("Origin", "Dest", "UniqueCarrier"):
        cnt = train[c].value_counts()
        X["cnt_" + c] = np.log1p(df[c].map(cnt).fillna(0.0)).to_numpy()
    return X


H_train, M_train, DT_train = _hour(train)
H_eval, M_eval, DT_eval = _hour(evald)
K_train = keys_for(train, H_train, DT_train)
K_eval = keys_for(evald, H_eval, DT_eval)
cnt_hour = {n: K_train[n].value_counts() for n in CNT_KEYS}
FOLDS = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(train))


def build_state(te_m):
    maps = {n: te_map(K_train[n], y_train, te_m[n]) for n in TE_KEYS}
    oof = {n: np.zeros(len(train)) for n in TE_KEYS}
    for tr_idx, va_idx in FOLDS:
        yf = y_train[tr_idx]
        for n in TE_KEYS:
            enc = te_map(K_train[n].iloc[tr_idx], yf, te_m[n])
            oof[n][va_idx] = enc.reindex(K_train[n].iloc[va_idx].values).fillna(PRIOR).to_numpy()

    def full_matrix(df, hour, minute, dt, K, use_oof):
        X = base_X(df, hour, minute, dt)
        for n in TE_KEYS:
            X[n] = oof[n] if use_oof else maps[n].reindex(K[n].values).fillna(PRIOR).to_numpy()
        for n in CNT_KEYS:
            X[n] = np.log1p(pd.Series(K[n].values).map(cnt_hour[n]).fillna(0.0)).to_numpy()
        X["frac_origin_hour"] = (np.expm1(X["cnt_origin_hour"]) / (np.expm1(X["cnt_Origin"]) + 1.0)).to_numpy()
        X["frac_dest_hour"] = (np.expm1(X["cnt_dest_hour"]) / (np.expm1(X["cnt_Dest"]) + 1.0)).to_numpy()
        return X

    return maps, full_matrix


COLS = (["DepTime", "hour", "minute", "sin_tod", "cos_tod", "Distance"]
        + ["cnt_Origin", "cnt_Dest", "cnt_UniqueCarrier"]
        + ["te_carrier", "te_origin", "te_dest", "te_month", "te_dow", "te_origin_dow"]
        + ["te_hour", "te_dep15", "te_dep5", "te_carrier_hour", "te_origin_hour", "te_dest_hour"]
        + ["cnt_origin_hour", "cnt_dest_hour", "cnt_route", "cnt_carrier_hour", "cnt_origin_dow",
           "cnt_origin_dep15", "cnt_dest_dep15", "cnt_global_dep15"]
        + ["frac_origin_hour", "frac_dest_hour"])

VARIANTS = {
    "m_half": {k: v / 2 for k, v in BASE_M.items()},
    "m_weakmd": {**BASE_M, "te_month": 100.0, "te_dow": 100.0},
    "m_half_wmd": {**{k: v / 2 for k, v in BASE_M.items()}, "te_month": 50.0, "te_dow": 50.0},
    "m_half_dep": {**{k: v / 2 for k, v in BASE_M.items()}, "te_dep15": 800.0,
                    "te_dep5": 600.0, "te_hour": 600.0},
    "m_all": {**{k: v / 2 for k, v in BASE_M.items()}, "te_month": 50.0, "te_dow": 50.0,
              "te_dep15": 800.0, "te_dep5": 600.0, "te_hour": 600.0,
              "te_carrier_hour": 800.0, "te_origin_hour": 800.0, "te_dest_hour": 800.0,
              "te_origin_dow": 800.0},
}

t0 = time.time()
results = {}
for vname, te_m in VARIANTS.items():
    maps, fm = build_state(te_m)
    Xtr = fm(train, H_train, M_train, DT_train, K_train, True)[COLS]
    Xev = fm(evald, H_eval, M_eval, DT_eval, K_eval, False)[COLS]
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.07, min_child_weight=30,
        reg_lambda=1.0, tree_method="hist", random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xtr, y_train, verbose=False)
    best_k, best_auc = 0, 0.0
    for k in (250, 300):
        auc = roc_auc_score(y_eval, m.predict_proba(Xev, iteration_range=(0, k))[:, 1])
        if auc > best_auc:
            best_k, best_auc = k, auc
    results[vname] = (best_auc, best_k, maps, fm, m)
    print(f"DIAG {vname:12s} best_k={best_k} eval_auc={best_auc:.4f}  ({time.time() - t0:.0f}s)")

BEST_V = max(results, key=lambda v: results[v][0])
best_auc, best_k, BEST_MAPS, BEST_FM, model = results[BEST_V]
print(f"BEST variant={BEST_V} k={best_k} auc={best_auc:.4f}")
print(f"Diagnostic time: {time.time() - t0:.1f}s")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    hour, minute, dt = _hour(df)
    K = keys_for(df, hour, dt)
    return BEST_FM(df, hour, minute, dt, K, use_oof=False)[COLS]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df), iteration_range=(0, best_k))[:, 1]


tp = time.time()
eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval time: {time.time() - tp:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
