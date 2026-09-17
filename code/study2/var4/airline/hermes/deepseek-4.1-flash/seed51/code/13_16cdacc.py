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


def _bucket_table(ids: np.ndarray, slots: np.ndarray) -> pd.Series:
    """count of training flights per (id, 15-min slot), keyed 'id_slot'."""
    uniq, inv = np.unique(ids, return_inverse=True)
    cnt = np.zeros((len(uniq), Q_PER_DAY), dtype=np.float64)
    np.add.at(cnt, (inv, slots), 1.0)
    keys = np.char.add(np.char.add(np.repeat(uniq, Q_PER_DAY), "_"),
                       np.tile(np.arange(Q_PER_DAY).astype(str), len(uniq)))
    return pd.Series(cnt.ravel(), index=keys)


def _rolling(table: pd.Series, width: int) -> pd.Series:
    """sum of `table` over a +/- `width` slot window (circular within the day)."""
    wide = table.to_numpy().reshape(-1, Q_PER_DAY)
    rolled = np.zeros_like(wide)
    for k in range(-width, width + 1):
        rolled += np.roll(wide, -k, axis=1)
    return pd.Series(rolled.ravel(), index=table.index)


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

# structural (target-free) description of the route network
NCAR_ROUTE = _train_keys.assign(carrier=train["UniqueCarrier"].astype(str)).groupby("route")["carrier"].nunique()
CARRIER_ROUTE_CNT = (_train_keys["route"] + "|" + train["UniqueCarrier"].astype(str)).value_counts()
del _train_keys, _dep_train, _origin_train, _dest_train, _dist_train, _slot_train, _arr_slot_train


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
    n_estimators=900,
    max_depth=6,
    learning_rate=0.025,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

MEMBER_OVERRIDES = [
    dict(random_state=SEED),
    dict(random_state=SEED + 1, subsample=0.7, colsample_bytree=0.6),
    dict(random_state=SEED + 2, max_depth=5, min_child_weight=10, n_estimators=1200),
    dict(random_state=SEED + 3, max_depth=8, min_child_weight=50, colsample_bytree=0.5),
    dict(random_state=SEED + 4, learning_rate=0.05, n_estimators=500, subsample=0.9),
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
