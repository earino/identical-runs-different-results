"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes
------------
The evaluation year (2006) differs from the training year (2005), so every engineered feature is a
*target-free* structural property of the flight network (scheduled slot of the day, airport/route traffic
volumes, carrier presence).  No encoders are fitted on the target: statistics that remember 2005 delay
rates do not transfer to 2006 and measurably hurt.
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

# --- raw columns ---------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# treat string columns as categoricals, but drop very high-cardinality ones (their splits do not transfer)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- flight-network keys (all target-free) --------------------------------------
Q_PER_DAY = 96          # 15-minute slots
SLOTS_PER_HOUR = 4


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _decode_c(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col].astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _clock(df: pd.DataFrame) -> pd.Series:
    """Scheduled departure as hhmm, with the hhmm+2400 next-day convention folded to 0-2359."""
    return _numeric(df, "DepTime").fillna(0).astype("int64") % 2400


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    dep = _clock(df)
    dist = _numeric(df, "Distance").fillna(0.0)
    hour = dep // 100
    half = hour * 2 + (dep % 100) // 30
    quarter = hour * 4 + (dep % 100) // 15
    # approximate slot at which this aircraft reaches its destination (~450 mph cruise, no turnaround):
    # a proxy for arrival-side congestion, which is what actually propagates delays
    arr_q = ((quarter + np.round(dist / 450.0 * 4)).astype("int64") % Q_PER_DAY).astype(str)
    quarter = quarter.astype(str)
    half = half.astype(str)
    hour = hour.astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    return pd.DataFrame({
        "origin": origin,
        "dest": dest,
        "carrier": carrier,
        "route": origin + "_" + dest,
        "carrier_origin": carrier + "_" + origin,
        "origin_hour": origin + "_" + hour,
        "dest_hour": dest + "_" + hour,
        "carrier_hour": carrier + "_" + hour,
        "route_hour": origin + "_" + dest + "_" + hour,
        "origin_half": origin + "_" + half,
        "dest_half": dest + "_" + half,
        "route_half": origin + "_" + dest + "_" + half,
        "origin_q": origin + "_" + quarter,
        "dest_q": dest + "_" + quarter,
        "route_q": origin + "_" + dest + "_" + quarter,
        "dest_arrq": dest + "_" + arr_q,
    })


def _bucket_table(ids: np.ndarray, slots: np.ndarray, n_slots: int = Q_PER_DAY) -> pd.Series:
    """count of training flights per (id, slot), keyed 'id_slot'."""
    uniq, inv = np.unique(ids, return_inverse=True)
    cnt = np.zeros((len(uniq), n_slots), dtype=np.float64)
    np.add.at(cnt, (inv, slots), 1.0)
    keys = np.char.add(np.char.add(np.repeat(uniq, n_slots), "_"),
                       np.tile(np.arange(n_slots).astype(str), len(uniq)))
    return pd.Series(cnt.ravel(), index=keys)


def _rolling(table: pd.Series, width: int, n_slots: int = Q_PER_DAY) -> pd.Series:
    """sum of `table` over a +/- `width` slot window (circular within the day)."""
    wide = table.to_numpy().reshape(-1, n_slots)
    rolled = np.zeros_like(wide)
    for k in range(-width, width + 1):
        rolled += np.roll(wide, -k, axis=1)
    return pd.Series(rolled.ravel(), index=table.index)


def _rolling_asym(table: pd.Series, back: int, fwd: int, n_slots: int = Q_PER_DAY) -> pd.Series:
    """sum of `table` over the trailing `back` and leading `fwd` slots (excluding the current slot)."""
    wide = table.to_numpy().reshape(-1, n_slots)
    out = np.zeros_like(wide)
    for k in range(-back, fwd + 1):
        if k != 0:
            out += np.roll(wide, -k, axis=1)
    return pd.Series(out.ravel(), index=table.index)


def _gaps(table: pd.Series, n_slots: int = Q_PER_DAY):
    """Schedule spacing, in slots: how long since / until the nearest flight of the same key."""
    wide = table.to_numpy().reshape(-1, n_slots)
    idx = np.tile(np.arange(n_slots, dtype=np.float64), (wide.shape[0], 1))
    seen = np.where(wide > 0, idx, -1.0)
    last = np.maximum.accumulate(seen, axis=1)
    back = np.where(last >= 0, idx - last, np.nan)
    nxt = np.where(wide > 0, idx, n_slots * 2.0)[:, ::-1]
    nxt = np.minimum.accumulate(nxt, axis=1)[:, ::-1]
    fwd = np.where(nxt < n_slots * 2.0, nxt - idx, np.nan)
    return (pd.Series(back.ravel(), index=table.index), pd.Series(fwd.ravel(), index=table.index))


_train_keys = _keys(train)
CNT_MAP = {k: _train_keys[k].value_counts() for k in _train_keys.columns}

_dep_train = _clock(train)
_origin_train = train["Origin"].astype(str).to_numpy()
_dest_train = train["Dest"].astype(str).to_numpy()
_dist_train = _numeric(train, "Distance").fillna(0.0).to_numpy()
_slot_train = (_dep_train // 100 * SLOTS_PER_HOUR + (_dep_train % 100) // 15).to_numpy()
_arr_slot_train = ((_slot_train + np.round(_dist_train / 450.0 * 4)).astype("int64") % Q_PER_DAY)

ORIGIN_15 = _rolling(_bucket_table(_origin_train, _slot_train), 1)
ORIGIN_30 = _rolling(_bucket_table(_origin_train, _slot_train), 2)
ORIGIN_60 = _rolling(_bucket_table(_origin_train, _slot_train), 4)
ARR_DEST_15 = _rolling(_bucket_table(_dest_train, _arr_slot_train), 1)
ARR_DEST_30 = _rolling(_bucket_table(_dest_train, _arr_slot_train), 2)
ORIGIN_IN_15 = _rolling(_bucket_table(_origin_train, _arr_slot_train), 1)
ORIGIN_IN_30 = _rolling(_bucket_table(_origin_train, _arr_slot_train), 2)
_carrier_origin_train = (train["UniqueCarrier"].astype(str) + "_" + train["Origin"].astype(str)).to_numpy()
CARRIER_ORIG_15 = _rolling(_bucket_table(_carrier_origin_train, _slot_train), 1)
CARRIER_ORIG_30 = _rolling(_bucket_table(_carrier_origin_train, _slot_train), 2)
_route_train = (_origin_train + "_" + _dest_train)
ROUTE_15 = _rolling(_bucket_table(_route_train, _slot_train), 1)
ROUTE_30 = _rolling(_bucket_table(_route_train, _slot_train), 2)

# 5-minute resolution: what matters for pushback queuing is the density right around the scheduled minute
S5 = 288
_slot5_train = (_dep_train // 100 * 12 + (_dep_train % 100) // 5).to_numpy()
ORIGIN5_15 = _rolling(_bucket_table(_origin_train, _slot5_train, S5), 3, S5)
ORIGIN5_30 = _rolling(_bucket_table(_origin_train, _slot5_train, S5), 6, S5)
CARRIER_ORIG5_15 = _rolling(_bucket_table(_carrier_origin_train, _slot5_train, S5), 3, S5)
CARRIER_ORIG5_30 = _rolling(_bucket_table(_carrier_origin_train, _slot5_train, S5), 6, S5)

# asymmetric windows: is this flight pushing back at the head or the tail of a departure bank?
ORIGIN_BACK_30 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 2, 0)
ORIGIN_FWD_30 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 0, 2)
CARRIER_ORIG_BACK_30 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 2, 0)
CARRIER_ORIG_FWD_30 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 0, 2)
ORIGIN_BACK_15 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 1, 0)
ORIGIN_FWD_15 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 0, 1)
ORIGIN_BACK_60 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 4, 0)
ORIGIN_FWD_60 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 0, 4)
CARRIER_ORIG_BACK_15 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 1, 0)
CARRIER_ORIG_FWD_15 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 0, 1)
CARRIER_ORIG_BACK_60 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 4, 0)
CARRIER_ORIG_FWD_60 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 0, 4)
ROUTE_BACK_30 = _rolling_asym(_bucket_table(_route_train, _slot_train), 2, 0)
ROUTE_FWD_30 = _rolling_asym(_bucket_table(_route_train, _slot_train), 0, 2)
ARR_DEST_BACK_30 = _rolling_asym(_bucket_table(_dest_train, _arr_slot_train), 2, 0)
ARR_DEST_FWD_30 = _rolling_asym(_bucket_table(_dest_train, _arr_slot_train), 0, 2)
ORIGIN_BACK_120 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 8, 0)
ORIGIN_FWD_120 = _rolling_asym(_bucket_table(_origin_train, _slot_train), 0, 8)
CARRIER_ORIG_BACK_120 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 8, 0)
CARRIER_ORIG_FWD_120 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 0, 8)
ROUTE_BACK_60 = _rolling_asym(_bucket_table(_route_train, _slot_train), 4, 0)
ROUTE_FWD_60 = _rolling_asym(_bucket_table(_route_train, _slot_train), 0, 4)
ROUTE_BACK_120 = _rolling_asym(_bucket_table(_route_train, _slot_train), 8, 0)
ROUTE_FWD_120 = _rolling_asym(_bucket_table(_route_train, _slot_train), 0, 8)
ARR_DEST_BACK_60 = _rolling_asym(_bucket_table(_dest_train, _arr_slot_train), 4, 0)
ARR_DEST_FWD_60 = _rolling_asym(_bucket_table(_dest_train, _arr_slot_train), 0, 4)
ARR_DEST_BACK_120 = _rolling_asym(_bucket_table(_dest_train, _arr_slot_train), 8, 0)
ARR_DEST_FWD_120 = _rolling_asym(_bucket_table(_dest_train, _arr_slot_train), 0, 8)
ORIGIN_BACK_ALL = _rolling_asym(_bucket_table(_origin_train, _slot_train), Q_PER_DAY - 1, 0)
ORIGIN_FWD_ALL = _rolling_asym(_bucket_table(_origin_train, _slot_train), 0, Q_PER_DAY - 1)
CARRIER_ORIG_BACK_ALL = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), Q_PER_DAY - 1, 0)
CARRIER_ORIG_FWD_ALL = _rolling_asym(_bucket_table(_carrier_origin_train, _slot_train), 0, Q_PER_DAY - 1)
ORIGIN5_BACK_15 = _rolling_asym(_bucket_table(_origin_train, _slot5_train, S5), 3, 0, S5)
ORIGIN5_FWD_15 = _rolling_asym(_bucket_table(_origin_train, _slot5_train, S5), 0, 3, S5)
ORIGIN5_BACK_30 = _rolling_asym(_bucket_table(_origin_train, _slot5_train, S5), 6, 0, S5)
ORIGIN5_FWD_30 = _rolling_asym(_bucket_table(_origin_train, _slot5_train, S5), 0, 6, S5)
CARRIER_ORIG5_BACK_15 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot5_train, S5), 3, 0, S5)
CARRIER_ORIG5_FWD_15 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot5_train, S5), 0, 3, S5)
CARRIER_ORIG5_BACK_30 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot5_train, S5), 6, 0, S5)
CARRIER_ORIG5_FWD_30 = _rolling_asym(_bucket_table(_carrier_origin_train, _slot5_train, S5), 0, 6, S5)

# schedule spacing around the flight (queue priority / how busy the surrounding schedule is)
_arr_slot5_train = ((_slot5_train + np.round(_dist_train / 450.0 * 12)).astype("int64") % S5)
ORIGIN_GAP_B, ORIGIN_GAP_F = _gaps(_bucket_table(_origin_train, _slot5_train, S5), S5)
CARRIER_GAP_B, CARRIER_GAP_F = _gaps(_bucket_table(_carrier_origin_train, _slot5_train, S5), S5)
ROUTE_GAP_B, ROUTE_GAP_F = _gaps(_bucket_table(_route_train, _slot5_train, S5), S5)
ARRDEST_GAP_B, ARRDEST_GAP_F = _gaps(_bucket_table(_dest_train, _arr_slot5_train, S5), S5)

# structural (target-free) description of the route network
NCAR_ROUTE = _train_keys.assign(carrier=train["UniqueCarrier"].astype(str)).groupby("route")["carrier"].nunique()
CARRIER_ROUTE_CNT = (_train_keys["route"] + "|" + train["UniqueCarrier"].astype(str)).value_counts()
del _train_keys, _dep_train, _origin_train, _dest_train, _dist_train, _slot_train, _arr_slot_train, _slot5_train


def _slot_keys(ids: np.ndarray, slots: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(ids, "_"), slots.astype(str))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    dep = _clock(df)
    hour = (dep // 100).astype("float64")
    minute = (dep % 100).astype("float64")
    X["dep_hour"] = hour
    X["dep_minute"] = minute
    X["dep_frac"] = hour + minute / 60.0
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_frac"] / 24.0)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_frac"] / 24.0)
    X["next_day"] = (_numeric(df, "DepTime").fillna(0) >= 2400).astype("int8")

    # ordinal duplicates of the seasonal columns: trees split monotonically on these, which transfers
    # better across years than the unordered categorical versions alone
    month = _decode_c(df, "Month")
    dom = _decode_c(df, "DayofMonth")
    dow = _decode_c(df, "DayOfWeek")
    X["month_num"] = month.astype("float64")
    X["dom_num"] = dom.astype("float64")
    X["dow_num"] = dow.astype("float64")
    X["month_sin"] = np.sin(2 * np.pi * X["month_num"] / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * X["month_num"] / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * X["dow_num"] / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * X["dow_num"] / 7.0)
    X["is_weekend"] = (X["dow_num"] >= 6).astype("int8")
    X["day_of_year"] = (X["month_num"] - 1) * 30.44 + X["dom_num"]

    dist = _numeric(df, "Distance")
    X["block_hours"] = dist / 450.0

    keys = _keys(df)
    for k in keys.columns:
        X[f"cnt_{k}"] = keys[k].map(CNT_MAP[k]).astype("float64").fillna(0.0)

    origin = df["Origin"].astype(str).to_numpy()
    dest = df["Dest"].astype(str).to_numpy()
    slot = (dep // 100 * SLOTS_PER_HOUR + (dep % 100) // 15).to_numpy()
    arr_slot = ((slot + np.round(dist.fillna(0.0).to_numpy() / 450.0 * 4)).astype("int64") % Q_PER_DAY)

    # local traffic density around the scheduled slot (a stronger congestion signal than the slot count)
    X["origin_15"] = pd.Series(_slot_keys(origin, slot)).map(ORIGIN_15).fillna(0.0).to_numpy()
    X["origin_30"] = pd.Series(_slot_keys(origin, slot)).map(ORIGIN_30).fillna(0.0).to_numpy()
    X["origin_60"] = pd.Series(_slot_keys(origin, slot)).map(ORIGIN_60).fillna(0.0).to_numpy()
    X["arrdest_15"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_15).fillna(0.0).to_numpy()
    X["arrdest_30"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_30).fillna(0.0).to_numpy()
    # inbound traffic landing at my origin while I am pushing back (turnaround pressure)
    X["origin_inbound"] = (
        pd.Series(_slot_keys(origin, slot)).map(CNT_MAP["dest_arrq"]).fillna(0.0).to_numpy()
    )
    X["origin_in_15"] = pd.Series(_slot_keys(origin, arr_slot)).map(ORIGIN_IN_15).fillna(0.0).to_numpy()
    X["origin_in_30"] = pd.Series(_slot_keys(origin, arr_slot)).map(ORIGIN_IN_30).fillna(0.0).to_numpy()
    carrier_origin = (df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)).to_numpy()
    X["carrier_orig_15"] = pd.Series(_slot_keys(carrier_origin, slot)).map(CARRIER_ORIG_15).fillna(0.0).to_numpy()
    X["carrier_orig_30"] = pd.Series(_slot_keys(carrier_origin, slot)).map(CARRIER_ORIG_30).fillna(0.0).to_numpy()
    route_arr = (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy()
    X["route_15"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_15).fillna(0.0).to_numpy()
    X["route_30"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_30).fillna(0.0).to_numpy()
    slot5 = (dep // 100 * 12 + (dep % 100) // 5).to_numpy()
    X["origin5_15"] = pd.Series(_slot_keys(origin, slot5)).map(ORIGIN5_15).fillna(0.0).to_numpy()
    X["origin5_30"] = pd.Series(_slot_keys(origin, slot5)).map(ORIGIN5_30).fillna(0.0).to_numpy()
    X["carrier_orig5_15"] = pd.Series(_slot_keys(carrier_origin, slot5)).map(CARRIER_ORIG5_15).fillna(0.0).to_numpy()
    X["carrier_orig5_30"] = pd.Series(_slot_keys(carrier_origin, slot5)).map(CARRIER_ORIG5_30).fillna(0.0).to_numpy()
    X["origin_back_30"] = pd.Series(_slot_keys(origin, slot)).map(ORIGIN_BACK_30).fillna(0.0).to_numpy()
    X["origin_fwd_30"] = pd.Series(_slot_keys(origin, slot)).map(ORIGIN_FWD_30).fillna(0.0).to_numpy()
    X["carrier_back_30"] = pd.Series(_slot_keys(carrier_origin, slot)).map(CARRIER_ORIG_BACK_30).fillna(0.0).to_numpy()
    X["carrier_fwd_30"] = pd.Series(_slot_keys(carrier_origin, slot)).map(CARRIER_ORIG_FWD_30).fillna(0.0).to_numpy()
    for tag, table_b, table_f in [
        ("15", ORIGIN_BACK_15, ORIGIN_FWD_15),
        ("60", ORIGIN_BACK_60, ORIGIN_FWD_60),
    ]:
        X[f"origin_back_{tag}"] = pd.Series(_slot_keys(origin, slot)).map(table_b).fillna(0.0).to_numpy()
        X[f"origin_fwd_{tag}"] = pd.Series(_slot_keys(origin, slot)).map(table_f).fillna(0.0).to_numpy()
    for tag, table_b, table_f in [
        ("15", CARRIER_ORIG_BACK_15, CARRIER_ORIG_FWD_15),
        ("60", CARRIER_ORIG_BACK_60, CARRIER_ORIG_FWD_60),
    ]:
        X[f"carrier_back_{tag}"] = pd.Series(_slot_keys(carrier_origin, slot)).map(table_b).fillna(0.0).to_numpy()
        X[f"carrier_fwd_{tag}"] = pd.Series(_slot_keys(carrier_origin, slot)).map(table_f).fillna(0.0).to_numpy()
    X["route_back_30"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_BACK_30).fillna(0.0).to_numpy()
    X["route_fwd_30"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_FWD_30).fillna(0.0).to_numpy()
    X["arrdest_back_30"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_BACK_30).fillna(0.0).to_numpy()
    X["arrdest_fwd_30"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_FWD_30).fillna(0.0).to_numpy()
    for tag, tb, tf in [("120", ORIGIN_BACK_120, ORIGIN_FWD_120)]:
        X[f"origin_back_{tag}"] = pd.Series(_slot_keys(origin, slot)).map(tb).fillna(0.0).to_numpy()
        X[f"origin_fwd_{tag}"] = pd.Series(_slot_keys(origin, slot)).map(tf).fillna(0.0).to_numpy()
    for tag, tb, tf in [("120", CARRIER_ORIG_BACK_120, CARRIER_ORIG_FWD_120)]:
        X[f"carrier_back_{tag}"] = pd.Series(_slot_keys(carrier_origin, slot)).map(tb).fillna(0.0).to_numpy()
        X[f"carrier_fwd_{tag}"] = pd.Series(_slot_keys(carrier_origin, slot)).map(tf).fillna(0.0).to_numpy()
    X["route_back_60"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_BACK_60).fillna(0.0).to_numpy()
    X["route_fwd_60"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_FWD_60).fillna(0.0).to_numpy()
    X["route_back_120"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_BACK_120).fillna(0.0).to_numpy()
    X["route_fwd_120"] = pd.Series(_slot_keys(route_arr, slot)).map(ROUTE_FWD_120).fillna(0.0).to_numpy()
    X["arrdest_back_60"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_BACK_60).fillna(0.0).to_numpy()
    X["arrdest_fwd_60"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_FWD_60).fillna(0.0).to_numpy()
    X["arrdest_back_120"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_BACK_120).fillna(0.0).to_numpy()
    X["arrdest_fwd_120"] = pd.Series(_slot_keys(dest, arr_slot)).map(ARR_DEST_FWD_120).fillna(0.0).to_numpy()
    X["origin_back_all"] = pd.Series(_slot_keys(origin, slot)).map(ORIGIN_BACK_ALL).fillna(0.0).to_numpy()
    X["origin_fwd_all"] = pd.Series(_slot_keys(origin, slot)).map(ORIGIN_FWD_ALL).fillna(0.0).to_numpy()
    X["carrier_back_all"] = pd.Series(_slot_keys(carrier_origin, slot)).map(CARRIER_ORIG_BACK_ALL).fillna(0.0).to_numpy()
    X["carrier_fwd_all"] = pd.Series(_slot_keys(carrier_origin, slot)).map(CARRIER_ORIG_FWD_ALL).fillna(0.0).to_numpy()
    for tag, tb, tf in [("15", ORIGIN5_BACK_15, ORIGIN5_FWD_15), ("30", ORIGIN5_BACK_30, ORIGIN5_FWD_30)]:
        X[f"origin5_back_{tag}"] = pd.Series(_slot_keys(origin, slot5)).map(tb).fillna(0.0).to_numpy()
        X[f"origin5_fwd_{tag}"] = pd.Series(_slot_keys(origin, slot5)).map(tf).fillna(0.0).to_numpy()
    for tag, tb, tf in [("15", CARRIER_ORIG5_BACK_15, CARRIER_ORIG5_FWD_15),
                        ("30", CARRIER_ORIG5_BACK_30, CARRIER_ORIG5_FWD_30)]:
        X[f"carrier5_back_{tag}"] = pd.Series(_slot_keys(carrier_origin, slot5)).map(tb).fillna(0.0).to_numpy()
        X[f"carrier5_fwd_{tag}"] = pd.Series(_slot_keys(carrier_origin, slot5)).map(tf).fillna(0.0).to_numpy()
    arr_slot5 = ((slot5 + np.round(dist.fillna(0.0).to_numpy() / 450.0 * 12)).astype("int64") % S5)
    X["origin_gap_back"] = pd.Series(_slot_keys(origin, slot5)).map(ORIGIN_GAP_B).to_numpy()
    X["origin_gap_fwd"] = pd.Series(_slot_keys(origin, slot5)).map(ORIGIN_GAP_F).to_numpy()
    X["carrier_gap_back"] = pd.Series(_slot_keys(carrier_origin, slot5)).map(CARRIER_GAP_B).to_numpy()
    X["carrier_gap_fwd"] = pd.Series(_slot_keys(carrier_origin, slot5)).map(CARRIER_GAP_F).to_numpy()
    X["route_gap_back"] = pd.Series(_slot_keys(route_arr, slot5)).map(ROUTE_GAP_B).to_numpy()
    X["route_gap_fwd"] = pd.Series(_slot_keys(route_arr, slot5)).map(ROUTE_GAP_F).to_numpy()
    X["arrdest_gap_back"] = pd.Series(_slot_keys(dest, arr_slot5)).map(ARRDEST_GAP_B).to_numpy()
    X["arrdest_gap_fwd"] = pd.Series(_slot_keys(dest, arr_slot5)).map(ARRDEST_GAP_F).to_numpy()

    X["ncar_route"] = keys["route"].map(NCAR_ROUTE).astype("float64").fillna(1.0)
    X["carrier_route_cnt"] = (
        keys["route"] + "|" + df["UniqueCarrier"].astype(str)
    ).map(CARRIER_ROUTE_CNT).astype("float64").fillna(0.0)

    cnt_origin = X["cnt_origin"].replace(0.0, np.nan)
    cnt_dest = X["cnt_dest"].replace(0.0, np.nan)
    X["route_share"] = X["cnt_route"] / cnt_origin
    X["route_dest_share"] = X["cnt_route"] / cnt_dest
    X["carrier_share"] = X["cnt_carrier_origin"] / cnt_origin
    X["origin_hour_share"] = X["cnt_origin_hour"] / cnt_origin
    X["origin_half_share"] = X["cnt_origin_half"] / cnt_origin
    X["origin_q_share"] = X["cnt_origin_q"] / cnt_origin
    X["dest_arr_share"] = X["cnt_dest_arrq"] / cnt_dest
    X["route_hour_share"] = X["cnt_route_hour"] / X["cnt_origin_hour"].replace(0.0, np.nan)
    X["carrier_route_share"] = X["carrier_route_cnt"] / X["cnt_route"].replace(0.0, np.nan)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# XGBoost has no seed-ensembling built in; averaging a few diverse boosters reduces variance, which is the
# dominant source of error for a 2005-trained model scored on 2006 rows.
BASE_PARAMS = dict(
    n_estimators=1300,
    max_depth=7,
    learning_rate=0.02,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

MEMBER_OVERRIDES = [
    dict(random_state=SEED),
    dict(random_state=SEED + 1, subsample=0.7, colsample_bytree=0.5),
    dict(random_state=SEED + 2, max_depth=5, min_child_weight=10, n_estimators=1800, colsample_bytree=0.7),
    dict(random_state=SEED + 3, max_depth=8, min_child_weight=60, colsample_bytree=0.45),
    dict(random_state=SEED + 4, learning_rate=0.04, n_estimators=650, subsample=0.9, max_depth=6),
]

X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
for ov in MEMBER_OVERRIDES:
    m = xgb.XGBClassifier(**{**BASE_PARAMS, **ov})
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
