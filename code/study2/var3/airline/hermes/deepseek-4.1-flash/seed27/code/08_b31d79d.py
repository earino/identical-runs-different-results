"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     Everything the model consumes is produced inside `prepare()`, and every statistic/encoder is fit on the
     TRAINING rows only, so `predict_proba` reproduces the same transformation on unseen rows.
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

# --- features -----------------------------------------------------------------
CAL_COLS = ["Month", "DayofMonth", "DayOfWeek"]      # stored as 'c-<n>' strings
FEATURE_COLS = [c for c in TASK["columns"] if c not in ID_COLS + [TARGET]]
CAT_COLS = [c for c in FEATURE_COLS if c not in CAL_COLS + ["DepTime", "Distance"]]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def cnum(s: pd.Series) -> pd.Series:
    """'c-12' -> 12."""
    return s.astype(str).str.slice(2).astype(int)


def dep_hour(df: pd.DataFrame) -> pd.Series:
    return (df["DepTime"].astype(int) // 100) % 24


def dep_minute(dep: pd.Series) -> pd.Series:
    """hhmm -> minutes since midnight (handles post-midnight codes such as 2620)."""
    d = dep.astype(int)
    return (d // 100) % 24 * 60 + d % 100


def _key(df: pd.DataFrame, cols, sep="|") -> pd.Series:
    s = df[cols[0]].astype(str)
    for c in cols[1:]:
        s = s + sep + df[c].astype(str)
    return s


def _sorted_times_by_key(src: pd.DataFrame, keyvals: np.ndarray, vals: np.ndarray = None) -> dict:
    """key -> sorted array of values (scheduled departure times) seen for that key in the TRAINING rows."""
    if vals is None:
        vals = src["DepTime"].values.astype(np.int64)
    s = pd.DataFrame({"k": keyvals, "t": vals}).sort_values(["k", "t"])
    ks, ts = s["k"].values, s["t"].values
    uniq, starts = np.unique(ks, return_index=True)
    ends = np.append(starts[1:], len(ks))
    return {u: ts[a:b] for u, a, b in zip(uniq, starts, ends)}


# --- schedule-density ("congestion") features ---------------------------------
# All of these are computed from the TRAINING rows only (no target involved) and applied to any frame by
# lookup, so they reproduce identically on unseen rows. They measure how busy an airport / route / carrier
# is at that scheduled hour, which carries over to later periods.
_h = dep_hour(train)
_hs = _h.astype(str)
CNT = {
    "orig_hour": _key(train, ["Origin"]) + "|" + _hs,
    "dest_hour": _key(train, ["Dest"]) + "|" + _hs,
    "route_hour": _key(train, ["Origin", "Dest"], "_") + "|" + _hs,
    "origin": train["Origin"].value_counts(),
    "dest": train["Dest"].value_counts(),
    "carrier_origin": _key(train, ["UniqueCarrier", "Origin"]),
    "carrier_hour": _key(train, ["UniqueCarrier"]) + "|" + _hs,
    "carrier_origin_hour": _key(train, ["UniqueCarrier", "Origin"]) + "|" + _hs,
}
for _k in ("orig_hour", "dest_hour", "route_hour", "carrier_origin", "carrier_hour", "carrier_origin_hour"):
    CNT[_k] = CNT[_k].value_counts()
LUT = {
    "orig_hour": _sorted_times_by_key(train, (_key(train, ["Origin"]) + "|" + _hs).values),
    "dest_hour": _sorted_times_by_key(train, (_key(train, ["Dest"]) + "|" + _hs).values),
    "route_hour": _sorted_times_by_key(train, (_key(train, ["Origin", "Dest"], "_") + "|" + _hs).values),
    "carrier_origin_hour": _sorted_times_by_key(train, (_key(train, ["UniqueCarrier", "Origin"]) + "|" + _hs).values),
}
# plain (hour-free) time lists, for sliding-window density in minutes since midnight
_dm_tr = dep_minute(train["DepTime"]).values.astype(np.int64)
LUT_MIN = {
    "origin": _sorted_times_by_key(train, _key(train, ["Origin"]).values, _dm_tr),
    "dest": _sorted_times_by_key(train, _key(train, ["Dest"]).values, _dm_tr),
    "route": _sorted_times_by_key(train, _key(train, ["Origin", "Dest"], "_").values, _dm_tr),
}


def _rank(df: pd.DataFrame, lut: dict, keyvals: np.ndarray) -> np.ndarray:
    """#training flights with the same key scheduled at or before this row's scheduled time (queue position)."""
    codes, uniq = pd.factorize(pd.Series(keyvals))
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    b = np.append(np.searchsorted(sc, np.arange(len(uniq))), len(codes))
    tv = df["DepTime"].values.astype(np.int64)
    out = np.zeros(len(df), dtype=float)
    for j, u in enumerate(uniq):
        idx = order[b[j]:b[j + 1]]
        arr = lut.get(u)
        if arr is not None:
            out[idx] = np.searchsorted(arr, tv[idx], side="left")
    return out


def _window(df: pd.DataFrame, lut: dict, keyvals: np.ndarray, minutes: int, side: str) -> np.ndarray:
    """#training flights with the same key scheduled within `minutes` before (or after) this row's time."""
    dm = dep_minute(df["DepTime"]).values.astype(np.int64)
    codes, uniq = pd.factorize(pd.Series(keyvals))
    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    b = np.append(np.searchsorted(sc, np.arange(len(uniq))), len(codes))
    out = np.zeros(len(df), dtype=float)
    for j, u in enumerate(uniq):
        idx = order[b[j]:b[j + 1]]
        arr = lut.get(u)
        if arr is None:
            continue
        t = dm[idx]
        if side == "before":
            out[idx] = np.searchsorted(arr, t, "left") - np.searchsorted(arr, t - minutes, "left")
        else:
            out[idx] = np.searchsorted(arr, t + minutes, "left") - np.searchsorted(arr, t, "left")
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in CAL_COLS:
        X[c] = cnum(df[c])
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    X["hour"] = (dep // 100) % 24
    X["minute"] = dep % 100
    X["dep_min"] = X["hour"] * 60 + X["minute"]
    X["Distance"] = df["Distance"].astype(float)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    h = X["hour"].astype(str)
    oh = (df["Origin"] + "|" + h).map(CNT["orig_hour"]).fillna(0.0).astype(float)
    dh = (df["Dest"] + "|" + h).map(CNT["dest_hour"]).fillna(0.0).astype(float)
    rh = (df["Origin"] + "_" + df["Dest"] + "|" + h).map(CNT["route_hour"]).fillna(0.0).astype(float)
    oc = df["Origin"].map(CNT["origin"]).fillna(0.0).astype(float)
    dc = df["Dest"].map(CNT["dest"]).fillna(0.0).astype(float)
    X["orig_hour_cnt"] = oh
    X["dest_hour_cnt"] = dh
    X["route_hour_cnt"] = rh
    X["orig_cnt"] = oc
    X["dest_cnt"] = dc
    X["orig_hour_share"] = oh / oc.replace(0.0, 1.0)
    X["carrier_origin_cnt"] = (df["UniqueCarrier"] + "|" + df["Origin"]).map(CNT["carrier_origin"]).fillna(0.0).astype(float)
    X["carrier_hour_cnt"] = (df["UniqueCarrier"] + "|" + h).map(CNT["carrier_hour"]).fillna(0.0).astype(float)
    X["orig_hour_before"] = _rank(df, LUT["orig_hour"], (df["Origin"] + "|" + h).values)
    X["dest_hour_before"] = _rank(df, LUT["dest_hour"], (df["Dest"] + "|" + h).values)
    X["route_hour_before"] = _rank(df, LUT["route_hour"], (df["Origin"] + "_" + df["Dest"] + "|" + h).values)
    _coh = df["UniqueCarrier"] + "|" + df["Origin"] + "|" + h
    X["carrier_origin_hour_cnt"] = _coh.map(CNT["carrier_origin_hour"]).fillna(0.0).astype(float)
    X["carrier_origin_hour_before"] = _rank(df, LUT["carrier_origin_hour"], _coh.values)
    X["origin_60_before"] = _window(df, LUT_MIN["origin"], df["Origin"].values, 60, "before")
    X["origin_60_after"] = _window(df, LUT_MIN["origin"], df["Origin"].values, 60, "after")
    X["origin_120_before"] = _window(df, LUT_MIN["origin"], df["Origin"].values, 120, "before")
    X["dest_60_before"] = _window(df, LUT_MIN["dest"], df["Dest"].values, 60, "before")
    X["route_120_before"] = _window(df, LUT_MIN["route"], (df["Origin"] + "_" + df["Dest"]).values, 120, "before")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=20,
    learning_rate=0.01,
    min_child_weight=1,
    reg_lambda=5.0,
    subsample=0.7,
    colsample_bytree=0.4,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
