"""XGBoost binary classifier for airline departure delay (dep_delayed_15min).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  All feature engineering lives inside prepare(), which predict_proba() applies to unseen rows too.
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

# --- feature schema (derived from TRAIN only; never from the df passed in) -----
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
NUM_COLS = [c for c in feature_cols if c not in cat_cols]
TIME_COLS = ["hour", "minute", "tod", "tod_sin", "tod_cos", "is_weekend", "is_red_eye", "log_dist",
             "origin_freq", "dest_freq", "route_freq", "carrier_freq",
             "route_tod_z", "carrier_tod_z", "origin_tod_z", "dest_tod_z",
             "route_hr_dens", "origin_hr_dens", "dest_hr_dens",
             "carrier_route_share", "carrier_origin_share", "carrier_dest_share",
             "route_pos01", "carrier_pos01", "origin_pos01", "route_is_first", "route_is_last"]
# train-only frequency maps for airports / routes / carriers (no target information)
_route_train = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str))
# schedule-shape stats (train only): where does this flight sit in its route's/carrier's day?
_tod_tr = (train["DepTime"].to_numpy(dtype=np.int64) % 2400 // 100 * 60
           + train["DepTime"].to_numpy(dtype=np.int64) % 2400 % 100).astype(np.float64)
_tod_stats = {}
for _name, _keys in (("route", _route_train), ("carrier", train["UniqueCarrier"].astype(str)),
                    ("origin", train["Origin"].astype(str)), ("dest", train["Dest"].astype(str))):
    _df = pd.DataFrame({"k": _keys.to_numpy(), "t": _tod_tr})
    _g = _df.groupby("k")["t"].agg(["mean", "std"])
    _tod_stats[_name] = (_g["mean"].to_dict(), _g["std"].fillna(0.0).to_dict())
# schedule density (train only): share of a route's / airport's flights in this departure hour
_hr_tr = (train["DepTime"].to_numpy(dtype=np.int64) % 2400 // 100)
_hour_dens = {}
for _name, _keys in (("route", _route_train), ("origin", train["Origin"].astype(str)),
                     ("dest", train["Dest"].astype(str))):
    _g = pd.DataFrame({"k": _keys.to_numpy(), "h": _hr_tr}).value_counts().rename("n").reset_index()
    _tot = _g.groupby("k")["n"].transform("sum")
    _hour_dens[_name] = dict(zip(zip(_g["k"], _g["h"]), _g["n"] / _tot))
# carrier dominance shares (train only)
_carr_tr = train["UniqueCarrier"].astype(str)
_shares = {}
for _name, _num, _den in (("carrier_route_share", _carr_tr + "|" + _route_train, _route_train),
                          ("carrier_origin_share", _carr_tr + "|" + train["Origin"].astype(str), train["Origin"].astype(str)),
                          ("carrier_dest_share", _carr_tr + "|" + train["Dest"].astype(str), train["Dest"].astype(str))):
    _n = _num.value_counts()
    _d = _den.value_counts()
    _shares[_name] = {k: float(v) / float(_d[k.split("|", 1)[1]]) for k, v in _n.items()}
# rotation position within the key's operational day (train only): first flight of the day is a
# fresh aircraft (rarely delayed), the last one absorbs the day's accumulated delay.
_pos = {}
for _name, _keys in (("route", _route_train), ("carrier", _carr_tr), ("origin", train["Origin"].astype(str))):
    _g = pd.DataFrame({"k": _keys.to_numpy(), "t": _tod_tr}).groupby("k")["t"].agg(["min", "max"])
    _pos[_name] = (_g["min"].to_dict(), _g["max"].to_dict())
_freq_maps = {
    "origin_freq": train["Origin"].value_counts(normalize=True).to_dict(),
    "dest_freq": train["Dest"].value_counts(normalize=True).to_dict(),
    "route_freq": _route_train.value_counts(normalize=True).to_dict(),
    "carrier_freq": train["UniqueCarrier"].value_counts(normalize=True).to_dict(),
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataframe (target may be absent) -> model matrix. Stateless: uses only train-derived maps."""
    X = df[NUM_COLS].copy()

    # scheduled departure time (hhmm int; 2400 means midnight)
    dt = df["DepTime"].to_numpy(dtype=np.int64) % 2400
    hour = dt // 100
    minute = dt % 100
    tod = (hour * 60 + minute).astype(np.float64)
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["is_red_eye"] = ((tod < 6 * 60) | (tod >= 22 * 60)).astype(np.int8)
    X["log_dist"] = np.log1p(df["Distance"].to_numpy(dtype=np.float64))

    dow = df["DayOfWeek"].str.slice(2).astype(int).to_numpy()
    X["is_weekend"] = (dow >= 6).astype(np.int8)

    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["origin_freq"] = df["Origin"].map(_freq_maps["origin_freq"]).fillna(0.0).to_numpy()
    X["dest_freq"] = df["Dest"].map(_freq_maps["dest_freq"]).fillna(0.0).to_numpy()
    X["route_freq"] = route.map(_freq_maps["route_freq"]).fillna(0.0).to_numpy()
    X["carrier_freq"] = df["UniqueCarrier"].map(_freq_maps["carrier_freq"]).fillna(0.0).to_numpy()

    for _name, _keys in (("route", route), ("carrier", df["UniqueCarrier"].astype(str)),
                         ("origin", df["Origin"].astype(str)), ("dest", df["Dest"].astype(str))):
        _m, _sd = _tod_stats[_name]
        _mean = _keys.map(_m)
        _std = _keys.map(_sd).replace(0.0, np.nan)
        X["%s_tod_z" % _name] = ((tod - _mean) / _std).fillna(0.0).to_numpy()

    for _name, _keys in (("route", route), ("origin", df["Origin"].astype(str)),
                         ("dest", df["Dest"].astype(str))):
        _map = _hour_dens[_name]
        X["%s_hr_dens" % _name] = [ _map.get((_k, _h), 0.0) for _k, _h in zip(_keys, hour) ]

    _carr = df["UniqueCarrier"].astype(str)
    for _col, _keys in (
        ("carrier_route_share", _carr + "|" + route),
        ("carrier_origin_share", _carr + "|" + df["Origin"].astype(str)),
        ("carrier_dest_share", _carr + "|" + df["Dest"].astype(str)),
    ):
        _map = _shares[_col]
        X[_col] = [_map.get(_k, 0.0) for _k in _keys]

    for _name, _keys in (("route", route), ("carrier", df["UniqueCarrier"].astype(str)),
                         ("origin", df["Origin"].astype(str))):
        _mn, _mx = _pos[_name]
        _lo = _keys.map(_mn).to_numpy()
        _hi = _keys.map(_mx).to_numpy()
        _span = np.where((_hi - _lo) > 0, _hi - _lo, 1.0)
        X["%s_pos01" % _name] = np.clip((tod - np.nan_to_num(_lo, nan=0.0)) / _span, 0, 1)
        if _name == "route":
            _tod_arr = np.asarray(tod)
            X["route_is_first"] = (_tod_arr <= np.nan_to_num(_lo, nan=-1e9)).astype(np.int8)
            X["route_is_last"] = (_tod_arr >= np.nan_to_num(_hi, nan=1e9)).astype(np.int8)

    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X[NUM_COLS + TIME_COLS + cat_cols]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: an ensemble of XGBoost models (seed/regularization bagging) --------
def make_model(seed: int, **over) -> xgb.XGBClassifier:
    p = dict(
        n_estimators=3000,
        max_depth=10,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.3,
        min_child_weight=80,
        reg_lambda=20.0,
        max_cat_threshold=64,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    p.update(over)
    return xgb.XGBClassifier(**p)


MODELS = [
    make_model(42, grow_policy="lossguide", max_depth=0, max_leaves=255, n_estimators=1500,
               learning_rate=0.017),
    make_model(202, grow_policy="lossguide", max_depth=0, max_leaves=255, n_estimators=1500,
               learning_rate=0.017, subsample=0.85, colsample_bytree=0.25),
    make_model(1337, grow_policy="lossguide", max_depth=0, max_leaves=127, n_estimators=2200,
               learning_rate=0.017, subsample=0.95, colsample_bytree=0.35, min_child_weight=60,
               reg_lambda=10.0),
]

X_TRAIN = prepare(train)
Y_TRAIN = to_y(train)
t0 = time.time()
for m in MODELS:
    m.fit(X_TRAIN, Y_TRAIN)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
