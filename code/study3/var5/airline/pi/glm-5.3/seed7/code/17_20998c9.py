"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame (target col may be absent) -> 1-D P(positive) array.
     ALL feature engineering lives inside prepare(); encoders are fit on training data only.

Notes (from experiments 1-2 + local diagnostics):
  - 2005 -> 2006 is a real distribution shift: models that memorize 2005-specific route stats
    (route Origin_Dest categorical, route target encoding) lose ~0.01 AUC on 2006.
  - Early stopping on a 2005 slice rewards MORE rounds and hurts 2006 AUC; use fixed, moderate
    capacity instead: depth 4, lr 0.05, 200 rounds.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- encoders: fit on TRAIN ONLY -------------------------------------------------
# NOTE: "Month" is deliberately NOT a categorical feature: month *identity* is
# year-specific (2005 month effects don't transfer to 2006); the smooth
# day-of-year + harmonics features carry the persistent seasonality instead.
CAT_COLS = ["DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}
MDAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
CUM = np.cumsum(np.concatenate([[0], MDAYS]))  # day-of-year via fixed non-leap calendar

# per-origin sorted departure minutes (schedule shape; fit on TRAIN only)
_dep_min_tr = train["DepTime"].astype(int).to_numpy() // 100 * 60 + train["DepTime"].astype(int).to_numpy() % 100
ECDF = {o: np.sort(g.to_numpy()) for o, g in pd.Series(_dep_min_tr).groupby(train["Origin"].astype(str).values)}
ECDF_DEST = {o: np.sort(g.to_numpy()) for o, g in pd.Series(_dep_min_tr).groupby(train["Dest"].astype(str).values)}
ECDF_CARRIER = {o: np.sort(g.to_numpy()) for o, g in pd.Series(_dep_min_tr).groupby(train["UniqueCarrier"].astype(str).values)}
ECDF_ORG_CARRIER = {
    o: np.sort(g.to_numpy())
    for o, g in pd.Series(_dep_min_tr).groupby((train["Origin"] + "|" + train["UniqueCarrier"]).astype(str).values)
}
ECDF_ROUTE = {
    o: np.sort(g.to_numpy())
    for o, g in pd.Series(_dep_min_tr).groupby((train["Origin"] + ">" + train["Dest"]).astype(str).values)
}
CARRIER_MEAN_DIST = train["Distance"].astype(float).groupby(train["UniqueCarrier"].astype(str).values).mean()
ORIGIN_N_CARRIERS = train.groupby(train["Origin"].astype(str).values)["UniqueCarrier"].nunique()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    t = df["DepTime"].astype(int)
    X["DepTime"] = t
    X["hour"] = t // 100
    X["dep_min"] = (t // 100) * 60 + t % 100
    X["minute_of_hour"] = t % 100
    mins = X["dep_min"].to_numpy()
    X["Distance"] = df["Distance"].astype(float)
    mo = df["Month"].str.slice(2).astype(int).to_numpy()
    da = df["DayofMonth"].str.slice(2).astype(int).to_numpy()
    doy = CUM[mo - 1] + da
    X["doy"] = doy
    X["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    # position of this flight within its origin/destination airport's day (congestion proxy)
    keys = df["Origin"].astype(str).to_numpy()
    ecdf = np.full(len(df), 0.5)
    for o in np.unique(keys):
        m = keys == o
        v = ECDF.get(o)
        if v is not None:
            ecdf[m] = np.searchsorted(v, mins[m]) / len(v)
    X["org_time_ecdf"] = ecdf
    keys_c = df["UniqueCarrier"].astype(str).to_numpy()
    X["dist_x_hour"] = X["Distance"].to_numpy() * (X["hour"].to_numpy() - 12)
    keys_d = df["Dest"].astype(str).to_numpy()
    ecdf_d = np.full(len(df), 0.5)
    for o in np.unique(keys_d):
        m = keys_d == o
        v = ECDF_DEST.get(o)
        if v is not None:
            ecdf_d[m] = np.searchsorted(v, mins[m]) / len(v)
    X["dst_time_ecdf"] = ecdf_d
    # asymmetric before-density: fraction of an airport's/carrier's daily
    # departures scheduled in the w minutes BEFORE this flight = the upstream
    # queue this departure inherits. Schedules are stable year over year, so this
    # is persistent structure, not a 2005 statistic. (After-side and symmetric
    # variants proved redundant in ablation.)
    keys_oc = keys + "|" + keys_c
    for w in (15, 30, 60, 90):
        X[f"org_before{w}"] = _dens_side(keys, ECDF, mins, w, before=True)
        X[f"dst_before{w}"] = _dens_side(keys_d, ECDF_DEST, mins, w, before=True)
    for w in (30, 60):
        X[f"car_before{w}"] = _dens_side(keys_c, ECDF_CARRIER, mins, w, before=True)
        X[f"orgcar_before{w}"] = _dens_side(keys_oc, ECDF_ORG_CARRIER, mins, w, before=True)
    # route-level schedule banks: a route's *schedule shape* is persistent year over
    # year (unlike route identity / outcome stats, which are year-specific and hurt)
    keys_r = keys + ">" + keys_d
    for w in (30, 60):
        X[f"route_dens{w}"] = _dens(keys_r, ECDF_ROUTE, mins, w)
        X[f"route_before{w}"] = _dens_side(keys_r, ECDF_ROUTE, mins, w, before=True)
    # exponential-decay queues (soft kernel): the upstream queue with smooth
    # time-decay; complements the hard-window before-densities
    for tau in (30, 60, 120):
        X[f"org_exp{tau}"] = _exp_queue(keys, ECDF, mins, tau)
        X[f"car_exp{tau}"] = _exp_queue(keys_c, ECDF_CARRIER, mins, tau)
        X[f"route_exp{tau}"] = _exp_queue(keys_r, ECDF_ROUTE, mins, tau)
    return X


def _group_slices(keys: np.ndarray):
    """factorize keys once, sort by code -> contiguous group slices (O(N log N))."""
    codes, uniq = pd.factorize(keys)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes[codes >= 0], minlength=len(uniq))
    ends = np.cumsum(counts)
    return order, ends, counts, uniq


def _dens(keys: np.ndarray, table: dict, mins: np.ndarray, w: int) -> np.ndarray:
    out = np.zeros(len(mins))
    order, ends, counts, uniq = _group_slices(keys)
    start = 0
    for gi in range(len(uniq)):
        end = ends[gi]
        if counts[gi] == 0:
            continue
        idx = order[start:end]
        v = table.get(str(uniq[gi]))
        if v is None or len(v) == 0:
            continue
        lo = np.searchsorted(v, mins[idx] - w, "left")
        hi = np.searchsorted(v, mins[idx] + w, "right")
        out[idx] = (hi - lo) / len(v)
        start = end
    return out


def _dens_side(keys: np.ndarray, table: dict, mins: np.ndarray, w: int, before: bool) -> np.ndarray:
    out = np.zeros(len(mins))
    order, ends, counts, uniq = _group_slices(keys)
    start = 0
    for gi in range(len(uniq)):
        end = ends[gi]
        if counts[gi] == 0:
            continue
        idx = order[start:end]
        v = table.get(str(uniq[gi]))
        if v is None or len(v) == 0:
            continue
        if before:
            lo = np.searchsorted(v, mins[idx] - w, "left")
            at = np.searchsorted(v, mins[idx], "right")
            out[idx] = (at - lo) / len(v)
        else:
            at = np.searchsorted(v, mins[idx], "left")
            hi = np.searchsorted(v, mins[idx] + w, "right")
            out[idx] = (hi - at) / len(v)
        start = end
    return out


def _exp_queue(keys: np.ndarray, table: dict, mins: np.ndarray, tau: float) -> np.ndarray:
    """Exponentially-decaying upstream queue: sum over earlier scheduled departures
    of exp(-(gap)/tau), normalized by group size. Soft-kernel version of the
    before-density: influence of a previous flight fades smoothly with its lead time."""
    out = np.zeros(len(mins))
    order, ends, counts, uniq = _group_slices(keys)
    start = 0
    for gi in range(len(uniq)):
        end = ends[gi]
        if counts[gi] == 0:
            continue
        idx = order[start:end]
        v = table.get(str(uniq[gi]))
        if v is None or len(v) == 0:
            continue
        m = mins[idx]
        w = np.exp(v / tau)
        csum = np.concatenate(([0.0], np.cumsum(w)))
        at = np.searchsorted(v, m, "right")
        out[idx] = np.exp(-m / tau) * csum[at] / len(v)
        start = end
    return out


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: fixed moderate capacity (no early stopping on a misleading 2005 slice).
# Small ensemble: colsample_bytree=0.7 gives the members real diversity; seeds alone do not.
N_MODELS = 5
t0 = time.time()
X_all = prepare(train)
y_all = to_y(train)
# evening flights (hour >= 16) are the hard, high-delay-rate segment: weight them up
sample_w = np.where(X_all["hour"].to_numpy() >= 15, 2.0, 1.0)
models = []
for s in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=7,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        random_state=s + 1,
        n_jobs=N_JOBS,
        min_child_weight=60,
        colsample_bytree=0.7,
        reg_alpha=1.0,
    )
    m.fit(X_all, y_all, sample_weight=sample_w)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
