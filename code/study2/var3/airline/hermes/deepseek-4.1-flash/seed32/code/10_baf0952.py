"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives inside prepare(), which predict_proba() calls on unseen rows.
Every encoder / category set / statistic is fitted on the TRAINING frame only; unseen keys fall back safely.

Feature story:
  * scheduled time of day is the dominant signal (delays accumulate through the day), so DepTime hhmm is
    decoded into hour / minute / minutes-of-day;
  * Month / DayofMonth / DayOfWeek ship as ordinal codes ('c-7'), so they are decoded to integers;
  * carrier / origin / dest stay native XGBoost categoricals (train-fixed level sets);
  * timetable-density counts (log1p of how many training flights share a route / airport / carrier, and, most
    importantly, share the same 15-minute departure slot, with smoothed 45/75-minute windows) act as a
    congestion proxy. These are target-free, so they transfer across the 2005 -> 2006 time shift, unlike
    target encodings and route identity features (both measured to hurt here).
"""
import json
import os
import re
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

# --- raw column roles (derived from the TRAINING frame only) --------------------
RAW_FEATURES = [c for c in train.columns if c not in ID_COLS + [TARGET]]

_C_TOKEN = re.compile(r"^c-(\d+)$")


def _is_str(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def _is_c_token(series: pd.Series) -> bool:
    """True if every non-null value looks like 'c-<int>' (an ordinal coded as a string)."""
    vals = series.dropna().unique()
    return len(vals) > 0 and all(_C_TOKEN.match(str(v)) for v in vals)


def _ord_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("^c-", "", regex=True), errors="coerce")


ORD_COLS = [c for c in RAW_FEATURES if _is_str(train[c]) and _is_c_token(train[c])]
CAT_COLS = [c for c in RAW_FEATURES if _is_str(train[c]) and c not in ORD_COLS
            and train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

DEPTIME = "DepTime" if "DepTime" in RAW_FEATURES else None


def _clock(df: pd.DataFrame) -> tuple:
    """DepTime hhmm integer -> (hour, minutes-of-day); 2400-style values wrapped to 0, junk -> NaN."""
    t = pd.to_numeric(df[DEPTIME], errors="coerce")
    t = t.where(t < 2400, t - 2400)
    h = np.floor(t / 100.0)
    m = t - h * 100.0
    ok = (h >= 0) & (h <= 23) & (m >= 0) & (m <= 59)
    hour = h.where((h >= 0) & (h <= 23), np.nan)
    return hour, (h * 60.0 + m).where(ok, np.nan)


def _str(series: pd.Series) -> pd.Series:
    return series.astype(str)


# --- timetable-density keys: log1p(count in training) per key -------------------
HAS_NETWORK = all(c in RAW_FEATURES for c in ("Origin", "Dest", "UniqueCarrier"))
_RT = lambda df: _str(df["Origin"]) + "_" + _str(df["Dest"])
_OR = lambda df: _str(df["Origin"])
_DE = lambda df: _str(df["Dest"])
_CA = lambda df: _str(df["UniqueCarrier"])
_H = lambda df: _clock(df)[0].astype("Int64").astype(str)
_B15 = lambda df: (_clock(df)[1] // 15).astype("Int64")

DENS_SPECS = {
    "route": _RT, "origin": _OR, "dest": _DE, "carrier": _CA,
    "origin_hour": lambda df: _OR(df) + "@" + _H(df),
    "dest_hour": lambda df: _DE(df) + "@" + _H(df),
    "carrier_hour": lambda df: _CA(df) + "@" + _H(df),
    "route_hour": lambda df: _RT(df) + "@" + _H(df),
    "origin_min15": lambda df: _OR(df) + "#" + _B15(df).astype(str),
    "dest_min15": lambda df: _DE(df) + "#" + _B15(df).astype(str),
    "route_min15": lambda df: _RT(df) + "#" + _B15(df).astype(str),
}
DENS = {k: {kk: float(np.log1p(vv)) for kk, vv in f(train).value_counts().items()}
        for k, f in DENS_SPECS.items()} if HAS_NETWORK else {}

# smoothed congestion windows: flights sharing a 15-min slot +/- 1 or +/- 2 slots
WINDOW_COLS = ["origin_win45", "dest_win45", "origin_win75", "route_win45"]


def _slot_table(entity_fn):
    tb = pd.DataFrame({"e": entity_fn(train), "b": _B15(train).astype("int64")})
    tb = tb.value_counts().reset_index(name="n")
    piv = tb.pivot_table(index="e", columns="b", values="n", fill_value=0)
    return piv.reindex(columns=range(96), fill_value=0)


def _lookup(piv, entity, bucket):
    idx = pd.MultiIndex.from_arrays([entity, bucket.astype("int64")])
    return piv.stack().reindex(idx).to_numpy()


if HAS_NETWORK:
    _piv_o, _piv_d, _piv_r = _slot_table(_OR), _slot_table(_DE), _slot_table(_RT)
    _roll = lambda p: p.rolling(window=3, min_periods=1, center=True, axis=1).sum()
    _roll75 = lambda p: p.rolling(window=5, min_periods=1, center=True, axis=1).sum()
    WIN_O45, WIN_D45, WIN_O75, WIN_R45 = _roll(_piv_o), _roll(_piv_d), _roll75(_piv_o), _roll(_piv_r)
    # the immediately preceding / following 15-minute slots at the same origin
    PREV_O15, NEXT_O15 = _piv_o.shift(1, axis=1), _piv_o.shift(-1, axis=1)
else:
    WIN_O45 = WIN_D45 = WIN_O75 = WIN_R45 = PREV_O15 = NEXT_O15 = None


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[RAW_FEATURES].copy()
    for c in ORD_COLS:
        X[c] = _ord_num(X[c])
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    if DEPTIME is not None:
        hour, mod = _clock(df)
        X["dep_hour"] = hour
        X["dep_minute"] = mod - hour * 60.0
        X["dep_minofday"] = mod
    if HAS_NETWORK:
        for name, spec in DENS_SPECS.items():
            X[name + "_cnt"] = spec(df).map(DENS[name]).fillna(0.0).to_numpy()
        b = _B15(df)
        for col, piv, ent in (("origin_win45", WIN_O45, _OR(df)), ("dest_win45", WIN_D45, _DE(df)),
                              ("origin_win75", WIN_O75, _OR(df)), ("route_win45", WIN_R45, _RT(df))):
            X[col] = np.nan_to_num(_lookup(piv, ent, b), nan=0.0)
        X["origin_min15_vs_hour"] = X["origin_min15_cnt"] - X["origin_hour_cnt"]
        X["dest_min15_vs_hour"] = X["dest_min15_cnt"] - X["dest_hour_cnt"]
        X["origin_prev15"] = np.nan_to_num(_lookup(PREV_O15, _OR(df), b), nan=0.0)
        X["origin_next15"] = np.nan_to_num(_lookup(NEXT_O15, _OR(df), b), nan=0.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of strongly regularized lossguide boosters -----------
PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.015,
    grow_policy="lossguide",
    max_depth=0,
    max_leaves=256,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.35,
    reg_lambda=20.0,
    max_bin=512,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
N_SEEDS = 3
X_TRAIN, Y_TRAIN = prepare(train), to_y(train)
MODELS = []
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(random_state=SEED + 100 * s, **PARAMS)
    t0 = time.time()
    m.fit(X_TRAIN, Y_TRAIN)
    print(f"Training time seed {s}: {time.time() - t0:.1f}s")
    MODELS.append(m)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
