"""XGBoost binary classifier for the airline task (see program.md).

Contract:
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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
# Baseline kept Month/DayOfWeek as c-<n> strings and DepTime as a raw hhmm integer.
# Both are poor encodings for a tree model: hhmm wraps around (2359 -> 0000) and the
# string levels carry no order. Decompose DepTime into hour/minute/time-of-day and turn
# the c-<n> columns into real numbers.
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
# category levels learned on TRAIN only; unseen levels at predict time become NaN.
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().astype(str).unique())) for c in CAT_COLS}


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _carrier_hour(df: pd.DataFrame) -> pd.Series:
    return df["UniqueCarrier"].astype(str) + "_" + df["DepTime"].astype(str).str.slice(0, 2)


def _hour_of(df: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(df["DepTime"], errors="coerce") // 100).clip(0, 23).astype("Int64").astype(str)


# --- calendar -----------------------------------------------------------------
# Only month/day/day-of-week are given (no year), so anything year-specific is out; these
# proxies use the month lengths of a non-leap year and weekday rules that land on the same
# holiday in both 2005 and 2006.
_CUM_DAYS = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)


def _doy(df: pd.DataFrame) -> pd.Series:
    m = pd.to_numeric(_cnum(df["Month"]), errors="coerce").clip(1, 12).fillna(1).astype(int).to_numpy()
    return pd.Series(_CUM_DAYS[m - 1] + _cnum(df["DayofMonth"]).fillna(0).to_numpy(), index=df.index)


def _week(df: pd.DataFrame) -> pd.Series:
    return (_doy(df) // 7).astype(int)


def _holiday(df: pd.DataFrame) -> pd.Series:
    m, d, w = _cnum(df["Month"]), _cnum(df["DayofMonth"]), _cnum(df["DayOfWeek"])
    h = ((m == 1) & (d <= 2)) | ((m == 7) & (d.between(3, 5))) | ((m == 12) & (d >= 20))
    h |= (m == 5) & (d >= 25) & (w == 1)          # Memorial Day
    h |= (m == 9) & (d <= 7) & (w == 1)           # Labor Day
    h |= (m == 11) & (d.between(22, 28)) & (w == 4)  # Thanksgiving
    return h.astype(int)


# Key builders for the aggregate features below.
KEY_FUNCS = {
    "route": _route,
    "carrier": lambda df: df["UniqueCarrier"].astype(str),
    "origin": lambda df: df["Origin"].astype(str),
    "dest": lambda df: df["Dest"].astype(str),
    "origin_hour": lambda df: df["Origin"].astype(str) + "_" + _hour_of(df),
    "dest_hour": lambda df: df["Dest"].astype(str) + "_" + _hour_of(df),
    "carrier_hour": _carrier_hour,
    "route_hour": lambda df: _route(df) + "_" + _hour_of(df),
    "carrier_origin": lambda df: df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str),
    "carrier_route": lambda df: df["UniqueCarrier"].astype(str) + "_" + _route(df),
}

# Target-encoded keys: interactions that XGBoost cannot split on directly without memorising
# levels, summarised by a smoothed target mean. Low-cardinality identities (carrier, origin,
# dest) are deliberately NOT encoded here — the native categoricals cover them, and adding
# the encoding on top measured worse. Seasonal keys (origin_month, dest_month, carrier_month,
# origin_dow, dest_dow) and carrier_route all measured worse too and were pruned.
TE_NAMES = ["route", "origin_hour", "dest_hour", "carrier_hour", "route_hour"]
TE_SPECS = {k: KEY_FUNCS[k] for k in TE_NAMES}
TE_K = 20.0
PRIOR = float((train[TARGET] == POSITIVE).mean())
FOLDS = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(len(train))))


def _te_lookup(key: pd.Series, y: np.ndarray) -> pd.Series:
    st = pd.DataFrame({"k": np.asarray(key), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (st["sum"] + PRIOR * TE_K) / (st["count"] + TE_K)


# full-train lookups used at predict time
TE_LUT = {name: _te_lookup(fn(train), (train[TARGET] == POSITIVE).astype(float).to_numpy()) for name, fn in TE_SPECS.items()}
# out-of-fold values the model is fitted on, so training rows never see their own target
TE_OOF = {}
for _name, _fn in TE_SPECS.items():
    _key = _fn(train)
    _y = (train[TARGET] == POSITIVE).astype(float).to_numpy()
    _oof = np.full(len(train), PRIOR)
    for _tr, _va in FOLDS:
        _lut = _te_lookup(_key.iloc[_tr], _y[_tr])
        _oof[_va] = _key.iloc[_va].map(_lut).fillna(PRIOR).to_numpy()
    TE_OOF[_name] = _oof


# Frequency of each key in train (log-scaled): how busy a route/airport-hour is, i.e. a
# congestion proxy, and a way to separate heavily-trafficked routes from rare ones.
# These added nothing at colsample 0.8 but help clearly at 0.2 (+0.0025).
COUNT_SPECS = {k: KEY_FUNCS[k] for k in ("route", "origin", "dest", "carrier", "origin_hour", "dest_hour", "carrier_hour")}
COUNT_LUT = {name: fn(train).value_counts() for name, fn in COUNT_SPECS.items()}

# How dominant a carrier is at an airport / on a route: log carrier-affinity, a hub proxy.
# Shares measured +0.0011 on eval; the raw count ratios are what matter, so only the log
# difference is fed to the model.
SHARE_SPECS = {"carrier_origin_share": ("carrier_origin", "origin"), "carrier_route_share": ("carrier_route", "route")}
SHARE_LUT = {n: (KEY_FUNCS[a](train).value_counts(), COUNT_LUT[b]) for n, (a, b) in SHARE_SPECS.items()}


# Cumulative traffic at an airport by the hour of day: how many flights (in train) are
# scheduled at or before this hour. Delays build up through the day, so this is the
# congestion the flight is departing into. Measured +0.0005 on eval; mechanism is sound.
def _cum_lut(side: str) -> pd.Series:
    counts = (train[side].astype(str) + "_" + _hour_of(train)).value_counts()
    by_ap: dict = {}
    for key, n in counts.items():
        ap, hr = str(key).rsplit("_", 1)
        by_ap.setdefault(ap, {})[int(hr)] = int(n)
    lut = {}
    for ap, hrs in by_ap.items():
        run = 0
        for hr in range(24):
            run += hrs.get(hr, 0)
            lut[f"{ap}_{hr}"] = run
    return pd.Series(lut)


CUM_LUT = {side: _cum_lut(side) for side in ("Origin", "Dest")}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    month = _cnum(df["Month"])
    dow = _cnum(df["DayOfWeek"])
    day = _cnum(df["DayofMonth"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 23)
    minute = dep % 100
    tod = hour * 60 + minute

    X["month"] = month
    X["day"] = day
    X["dow"] = dow
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["is_weekend"] = (dow >= 6).astype(int)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["doy"] = _doy(df)
    X["week"] = _week(df)
    X["is_holiday"] = _holiday(df)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=CAT_LEVELS[c])
    for name, fn in TE_SPECS.items():
        X[f"te_{name}"] = fn(df).map(TE_LUT[name]).fillna(PRIOR).astype(float)
    for name, fn in COUNT_SPECS.items():
        X[f"cnt_{name}"] = np.log1p(fn(df).map(COUNT_LUT[name]).fillna(0).to_numpy())
    for name, (num_key, den_key) in SHARE_SPECS.items():
        num_lut, den_lut = SHARE_LUT[name]
        X[name] = np.log1p(KEY_FUNCS[num_key](df).map(num_lut).fillna(0).to_numpy()) - np.log1p(
            KEY_FUNCS[den_key](df).map(den_lut).fillna(0).to_numpy()
        )
    for side, lut in CUM_LUT.items():
        X[f"cum_{side.lower()}"] = np.log1p((df[side].astype(str) + "_" + _hour_of(df)).map(lut).fillna(0).to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=500,
    max_depth=14,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.2,
    min_child_weight=1,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)
model = xgb.XGBClassifier(**PARAMS)

# Deep trees (max_depth 14) measured clearly better on eval than the shallow default at every
# depth tried (6 -> 14 monotonically improved): the interactions between identity, hour and the
# aggregate encodings are what generalises to the next year, so the model wants the capacity.
# A small bag over seeds removes some seed-to-seed variance; 4 models keeps the run at ~70s,
# well inside the 120s experiment limit.
SEEDS = [42, 7, 2024, 31337]
t0 = time.time()
X_train = prepare(train)
for name in TE_SPECS:
    X_train[f"te_{name}"] = TE_OOF[name]  # out-of-fold values for the fitted rows
y_train = to_y(train)
models = []
for sd in SEEDS:
    m = xgb.XGBClassifier(**{**PARAMS, "random_state": sd})
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
