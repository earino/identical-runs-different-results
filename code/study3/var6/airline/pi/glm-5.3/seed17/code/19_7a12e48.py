"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}

# --- schedule statistics (fit on train only, no target involved) ---------------
_depnum = pd.to_numeric(train["DepTime"], errors="coerce")
_hour_tr = (_depnum // 100).fillna(-1).astype(int).astype(str)
_hb_key_tr = (_depnum // 100 // 3).fillna(-1).astype(int).astype(str)
_depmin_tr = (_depnum // 100 * 60 + _depnum % 100)
# estimated scheduled arrival: departure + taxi/climb + distance at ~480 mph
_arr_min_tr = (_depmin_tr + 35.0 + pd.to_numeric(train["Distance"], errors="coerce") / 8.0) % 1440.0
_arr_hour_tr = (_arr_min_tr // 60).fillna(-1).astype(int).astype(str)

VOL_MAPS = {
    "origin_hour": (train["Origin"].astype(str) + "_" + _hour_tr).value_counts().to_dict(),
    "dest_hour": (train["Dest"].astype(str) + "_" + _hour_tr).value_counts().to_dict(),
    "origin_day": train["Origin"].astype(str).value_counts().to_dict(),
    "dest_day": train["Dest"].astype(str).value_counts().to_dict(),
    "route_day": (train["Origin"].astype(str) + "-" + train["Dest"].astype(str)).value_counts().to_dict(),
    "carrier_hour": (train["UniqueCarrier"].astype(str) + "_" + _hour_tr).value_counts().to_dict(),
    "origin_hb": (train["Origin"].astype(str) + "_" + _hb_key_tr).value_counts().to_dict(),
    "dest_hb": (train["Dest"].astype(str) + "_" + _hb_key_tr).value_counts().to_dict(),
    "dest_arr_hour": (train["Dest"].astype(str) + "_" + _arr_hour_tr).value_counts().to_dict(),
}

# operating window (minutes from midnight) per airport, from train scheduled times
_win = {}
for port_col, mins in [("Origin", _depmin_tr), ("Dest", _arr_min_tr)]:
    g = mins.groupby(train[port_col]).quantile([0.05, 0.95]).unstack()
    _win[port_col] = {k: (float(v[0.05]), max(float(v[0.95]) - float(v[0.05]), 60.0)) for k, v in g.iterrows()}

# sorted per-airport minute arrays for smooth windowed congestion density (train only)
_dep_sorted = {p: np.sort(v.to_numpy()) for p, v in _depmin_tr.groupby(train["Origin"])}
_arr_sorted = {p: np.sort(v.to_numpy()) for p, v in _arr_min_tr.groupby(train["Dest"])}
# arrivals INTO an airport (keyed by the airport they land at) = incoming aircraft supply
_arr_into = {p: np.sort(v.to_numpy()) for p, v in _arr_min_tr.groupby(train["Dest"])}
# same-carrier banks at an origin
_carrier_origin_key = train["UniqueCarrier"].astype(str) + "_" + train["Origin"].astype(str)
_bank_sorted = {p: np.sort(v.to_numpy()) for p, v in _depmin_tr.groupby(_carrier_origin_key)}


def _port_blocks(ports: np.ndarray):
    codes, uniq = pd.factorize(ports)
    order = np.argsort(codes, kind="stable")
    cs = codes[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]) if len(cs) else np.array([0], dtype=int)
    return order, cs, starts, uniq


def _window_density(blocks, minutes: np.ndarray, table: dict, lo: float = 60.0, hi: float = 60.0) -> np.ndarray:
    order, cs, starts, uniq = blocks
    out = np.zeros(len(order), dtype=float)
    ends = np.r_[starts[1:], len(cs)]
    for s, e in zip(starts, ends):
        arr = table.get(uniq[cs[s]])
        if arr is None or len(arr) == 0:
            continue
        vals = minutes[order[s:e]]
        out[order[s:e]] = np.searchsorted(arr, vals + hi) - np.searchsorted(arr, vals - lo)
    return out


def _cnum(s: pd.Series) -> pd.Series:
    """c-<n> string column -> numeric (leave NaN as NaN)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    X["month"] = month
    X["day"] = _cnum(df["DayofMonth"])
    dow = _cnum(df["DayOfWeek"])
    X["dow"] = dow
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    X["dep_time"] = dep
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_dist"] = np.log1p(dist)
    hour = (dep // 100).fillna(0).astype(int)
    minute = (dep % 100).fillna(0).astype(int)
    X["hour"] = hour
    X["minute"] = minute
    depmin = hour * 60 + minute
    X["depmin"] = depmin
    tod = depmin / 1440.0
    X["sin_tod"] = np.sin(2 * np.pi * tod)
    X["cos_tod"] = np.cos(2 * np.pi * tod)
    # phase-aligned day cycle: 0 at ~5am, the daily delay minimum (delay accumulates through the operating day)
    dmin = (depmin - 300) % 1440
    X["depmin_shift"] = dmin
    tod2 = dmin / 1440.0
    X["sin_tod_shift"] = np.sin(2 * np.pi * tod2)
    X["cos_tod_shift"] = np.cos(2 * np.pi * tod2)
    X["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    X["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    X["sin_dow"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["cos_dow"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    # schedule-volume statistics from TRAIN 2005 (stable structural facts, no target involved)
    oh = df["Origin"].astype(str) + "_" + hour.astype(str)
    dh = df["Dest"].astype(str) + "_" + hour.astype(str)
    hb = (hour // 3).astype(str)
    ohb = df["Origin"].astype(str) + "_" + hb
    dhb = df["Dest"].astype(str) + "_" + hb
    X["vol_origin_hour"] = oh.map(VOL_MAPS["origin_hour"]).fillna(0)
    X["vol_dest_hour"] = dh.map(VOL_MAPS["dest_hour"]).fillna(0)
    X["vol_origin_day"] = df["Origin"].astype(str).map(VOL_MAPS["origin_day"]).fillna(0)
    X["vol_dest_day"] = df["Dest"].astype(str).map(VOL_MAPS["dest_day"]).fillna(0)
    X["vol_route_day"] = (df["Origin"].astype(str) + "-" + df["Dest"].astype(str)).map(VOL_MAPS["route_day"]).fillna(0)
    X["vol_carrier_hour"] = (df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)).map(VOL_MAPS["carrier_hour"]).fillna(0)
    X["vol_origin_hb"] = ohb.map(VOL_MAPS["origin_hb"]).fillna(0)
    X["vol_dest_hb"] = dhb.map(VOL_MAPS["dest_hb"]).fillna(0)
    X["share_origin_hour"] = X["vol_origin_hour"] / (X["vol_origin_day"] + 1)
    X["share_dest_hour"] = X["vol_dest_hour"] / (X["vol_dest_day"] + 1)
    # estimated arrival congestion + relative position in the airport's operating day
    est_arr = (depmin + 35.0 + dist / 8.0) % 1440.0
    arr_hour = (est_arr // 60).fillna(-1).astype(int).astype(str)
    X["vol_dest_arr"] = (df["Dest"].astype(str) + "_" + arr_hour).map(VOL_MAPS["dest_arr_hour"]).fillna(0)
    X["share_dest_arr"] = X["vol_dest_arr"] / (X["vol_dest_day"] + 1)
    arr_tod = est_arr / 1440.0
    X["sin_arr"] = np.sin(2 * np.pi * arr_tod)
    X["cos_arr"] = np.cos(2 * np.pi * arr_tod)
    # windowed congestion densities around this flight's times, from train schedules
    ob = _port_blocks(df["Origin"].to_numpy())
    db = _port_blocks(df["Dest"].to_numpy())
    depmin_np = depmin.to_numpy(dtype=float)
    arr_np = est_arr.to_numpy(dtype=float)
    X["cong_origin_dep"] = _window_density(ob, depmin_np, _dep_sorted)
    X["cong_dest_arr"] = _window_density(db, arr_np, _arr_sorted)
    X["cong_origin_before"] = _window_density(ob, depmin_np, _dep_sorted, 90.0, 0.0)
    X["cong_origin_after"] = _window_density(ob, depmin_np, _dep_sorted, 0.0, 90.0)
    X["cong_dest_before"] = _window_density(db, arr_np, _arr_sorted, 90.0, 0.0)
    X["cong_origin_120"] = _window_density(ob, depmin_np, _dep_sorted, 120.0, 120.0)
    X["intens_origin"] = X["cong_origin_dep"] / (X["vol_origin_day"] + 1)
    X["intens_dest"] = X["cong_dest_arr"] / (X["vol_dest_day"] + 1)
    # fine-grained queues and forward arrivals at the destination
    X["cong_origin_dep30"] = _window_density(ob, depmin_np, _dep_sorted, 30.0, 30.0)
    X["cong_dest_arr30"] = _window_density(db, arr_np, _arr_sorted, 30.0, 30.0)
    X["cong_dest_after"] = _window_density(db, arr_np, _arr_sorted, 0.0, 90.0)
    X["cong_origin_dep15"] = _window_density(ob, depmin_np, _dep_sorted, 15.0, 15.0)
    X["cong_dest_arr15"] = _window_density(db, arr_np, _arr_sorted, 15.0, 15.0)
    X["cong_incoming15"] = _window_density(ob, depmin_np, _arr_into, 30.0, 0.0)
    X["cong_origin_m60p120"] = _window_density(ob, depmin_np, _dep_sorted, 60.0, 120.0)
    # incoming aircraft supply: train flights landing at this ORIGIN shortly before departure
    X["cong_incoming"] = _window_density(ob, depmin_np, _arr_into, 90.0, 0.0)
    X["cong_incoming_wide"] = _window_density(ob, depmin_np, _arr_into, 180.0, 0.0)
    # same-carrier bank congestion at the origin
    bank_key = df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    bb = _port_blocks(bank_key.to_numpy())
    X["cong_carrier_bank"] = _window_density(bb, depmin_np, _bank_sorted)
    X["share_carrier_bank"] = X["cong_carrier_bank"] / (X["vol_origin_day"] + 1)
    o_win = df["Origin"].map(lambda p: _win["Origin"].get(p, (350.0, 800.0)))
    X["rel_pos_origin"] = ((depmin - o_win.map(lambda t: t[0])) / o_win.map(lambda t: t[1])).clip(-0.2, 1.3)
    d_win = df["Dest"].map(lambda t: _win["Dest"].get(t, (350.0, 800.0)))
    X["rel_pos_dest"] = ((est_arr - d_win.map(lambda t: t[0])) / d_win.map(lambda t: t[1])).clip(-0.2, 1.3)
    # native categoricals ablated this experiment: volume features carry airport/carrier identity
    return X

def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# ensemble of diverse XGB configs (each run twice with different seeds); average their probabilities.
CONFIGS = [
    dict(n_estimators=200, max_depth=8, learning_rate=0.1, colsample_bytree=0.8, subsample=0.8, min_child_weight=10),
    dict(n_estimators=400, max_depth=6, learning_rate=0.05, colsample_bytree=0.7, subsample=0.9, min_child_weight=10),
    dict(n_estimators=120, max_depth=10, learning_rate=0.1, colsample_bytree=0.9, subsample=0.7, min_child_weight=10),
    dict(n_estimators=800, max_depth=4, learning_rate=0.03, colsample_bytree=0.8, subsample=0.85, min_child_weight=10),
    dict(n_estimators=300, max_depth=7, learning_rate=0.07, colsample_bytree=0.6, subsample=0.75, min_child_weight=25),
    dict(n_estimators=80, max_depth=12, learning_rate=0.15, colsample_bytree=1.0, subsample=0.8, min_child_weight=5),
    dict(n_estimators=250, max_depth=9, learning_rate=0.1, colsample_bytree=0.75, subsample=0.75, min_child_weight=15),
    dict(n_estimators=600, max_depth=5, learning_rate=0.04, colsample_bytree=0.7, subsample=0.9, min_child_weight=20),
]
X_all, y_all = prepare(train), to_y(train)
models = []
t0 = time.time()
for i, cfg in enumerate(CONFIGS * 3):  # each config 3x with different seeds
    m = xgb.XGBClassifier(
        **cfg,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 7 * i,  # distinct seed per member
        n_jobs=N_JOBS,
    )
    m.fit(X_all, y_all)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")
imp = pd.Series(models[0].feature_importances_, index=X_all.columns).sort_values(ascending=False)
print("feature importance (model 0):\n" + imp.round(4).to_string())


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    Ps = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(Ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
