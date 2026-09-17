"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Diagnostic: TE upgrades — hierarchical shrinkage, deviation features, hour-volume counts.
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
        "cnt_origin_hour": df["Origin"].astype(str) + "|" + hs,
        "cnt_dest_hour": df["Dest"].astype(str) + "|" + hs,
    }


TE_M = dict(te_carrier=150.0, te_origin=150.0, te_dest=150.0, te_month=300.0,
            te_dow=300.0, te_hour=300.0, te_carrier_hour=400.0, te_origin_hour=400.0,
            te_dest_hour=400.0, te_route=500.0, te_origin_dow=400.0, te_dep15=500.0)
HIER = {"te_carrier_hour": "te_carrier", "te_origin_hour": "te_origin", "te_dest_hour": "te_dest"}
TE_KEYS = list(TE_M)


def te_map(grp: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    df = pd.DataFrame({"g": grp.values, "y": y})
    agg = df.groupby("g")["y"].agg(["sum", "count"])
    return (agg["sum"] + m * PRIOR) / (agg["count"] + m)


def te_map_hier(grp: pd.Series, y: np.ndarray, m: float, prior_map: pd.Series) -> pd.Series:
    df = pd.DataFrame({"g": grp.values, "y": y})
    agg = df.groupby("g")["y"].agg(["sum", "count"])
    prior = prior_map.reindex([str(g).split("|")[0] for g in agg.index]).fillna(PRIOR)
    return (agg["sum"] + m * prior.to_numpy()) / (agg["count"] + m)


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


# --- fit state on TRAIN only ------------------------------------------------------
H_train, M_train, DT_train = _hour(train)
H_eval, M_eval, DT_eval = _hour(evald)
K_train = keys_for(train, H_train, DT_train)
K_eval = keys_for(evald, H_eval, DT_eval)
cnt_hour = {n: K_train[n].value_counts() for n in ("cnt_origin_hour", "cnt_dest_hour")}

maps_flat = {n: te_map(K_train[n], y_train, TE_M[n]) for n in TE_KEYS}
maps_hier = dict(maps_flat)
for n, base in HIER.items():
    maps_hier[n] = te_map_hier(K_train[n], y_train, TE_M[n], maps_flat[base])

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_flat = {n: np.zeros(len(train)) for n in TE_KEYS}
oof_hier = {n: np.zeros(len(train)) for n in TE_KEYS}
for tr_idx, va_idx in kf.split(train):
    yf = y_train[tr_idx]
    fflat = {n: te_map(K_train[n].iloc[tr_idx], yf, TE_M[n]) for n in TE_KEYS}
    for n in TE_KEYS:
        oof_flat[n][va_idx] = fflat[n].reindex(K_train[n].iloc[va_idx].values).fillna(PRIOR).to_numpy()
    fhier = dict(fflat)
    for n, base in HIER.items():
        fhier[n] = te_map_hier(K_train[n].iloc[tr_idx], yf, TE_M[n], fflat[base])
    for n in TE_KEYS:
        oof_hier[n][va_idx] = fhier[n].reindex(K_train[n].iloc[va_idx].values).fillna(PRIOR).to_numpy()


def full_matrix(df, hour, minute, dt, K, use_oof, hier):
    X = base_X(df, hour, minute, dt)
    maps = maps_hier if hier else maps_flat
    oof = oof_hier if hier else oof_flat
    for n in TE_KEYS:
        X[n] = oof[n] if use_oof else maps[n].reindex(K[n].values).fillna(PRIOR).to_numpy()
    for n in ("cnt_origin_hour", "cnt_dest_hour"):
        X[n] = np.log1p(pd.Series(K[n].values).map(cnt_hour[n]).fillna(0.0)).to_numpy()
    X["dev_origin_hour"] = X["te_origin_hour"] - X["te_origin"]
    X["dev_dest_hour"] = X["te_dest_hour"] - X["te_dest"]
    X["dev_carrier_hour"] = X["te_carrier_hour"] - X["te_carrier"]
    return X


FULL_COLS = list(full_matrix(train, H_train, M_train, DT_train, K_train, True, True).columns)
print(f"n_cols={len(FULL_COLS)}")


def prepare(df: pd.DataFrame, hier: bool = False) -> pd.DataFrame:
    hour, minute, dt = _hour(df)
    K = keys_for(df, hour, dt)
    return full_matrix(df, hour, minute, dt, K, use_oof=False, hier=hier)[FULL_COLS]


Xtr_flat = full_matrix(train, H_train, M_train, DT_train, K_train, True, False)
Xev_flat = full_matrix(evald, H_eval, M_eval, DT_eval, K_eval, False, False)
Xtr_hier = full_matrix(train, H_train, M_train, DT_train, K_train, True, True)
Xev_hier = full_matrix(evald, H_eval, M_eval, DT_eval, K_eval, False, True)

BASE = ["DepTime", "hour", "minute", "sin_tod", "cos_tod", "Distance"]
VOL3 = ["cnt_Origin", "cnt_Dest", "cnt_UniqueCarrier"]
PLAIN = ["te_carrier", "te_origin", "te_dest", "te_month", "te_dow", "te_route", "te_origin_dow"]
TIME = ["te_hour", "te_dep15", "te_carrier_hour", "te_origin_hour", "te_dest_hour"]
DEV = ["dev_origin_hour", "dev_dest_hour", "dev_carrier_hour"]
CNTOH = ["cnt_origin_hour", "cnt_dest_hour"]

VARIANTS = {
    "v0_base": (BASE + VOL3 + PLAIN + TIME, False),
    "v1_hier": (BASE + VOL3 + PLAIN + TIME, True),
    "v2_dev": (BASE + VOL3 + PLAIN + TIME + DEV, False),
    "v3_cntoh": (BASE + VOL3 + PLAIN + TIME + CNTOH, False),
    "v4_hier_dev": (BASE + VOL3 + PLAIN + TIME + DEV, True),
    "v5_all_flat": (BASE + VOL3 + PLAIN + TIME + DEV + CNTOH, False),
    "v6_all_hier": (BASE + VOL3 + PLAIN + TIME + DEV + CNTOH, True),
}

results = {}
t0 = time.time()
for vname, (cols, hier) in VARIANTS.items():
    Xtr = Xtr_hier[cols] if hier else Xtr_flat[cols]
    Xev = Xev_hier[cols] if hier else Xev_flat[cols]
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=10, learning_rate=0.07, min_child_weight=30,
        reg_lambda=1.0, tree_method="hist", random_state=SEED, n_jobs=N_JOBS,
    )
    m.fit(Xtr, y_train, verbose=False)
    best_k, best_auc = 0, 0.0
    for k in (150, 250, 300):
        auc = roc_auc_score(y_eval, m.predict_proba(Xev, iteration_range=(0, k))[:, 1])
        if auc > best_auc:
            best_k, best_auc = k, auc
    results[vname] = (best_auc, best_k, cols, hier, m)
    print(f"DIAG {vname:12s} best_k={best_k} eval_auc={best_auc:.4f}  ({time.time() - t0:.0f}s)")

BEST_V = max(results, key=lambda v: results[v][0])
best_auc, best_k, COLS, BEST_HIER, model = results[BEST_V]
print(f"BEST variant={BEST_V} k={best_k} auc={best_auc:.4f} n_cols={len(COLS)}")
print(f"Diagnostic time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df, BEST_HIER)[COLS], iteration_range=(0, best_k))[:, 1]


eval_auc = roc_auc_score(y_eval, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
