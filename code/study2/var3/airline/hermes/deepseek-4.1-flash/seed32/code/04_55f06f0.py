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
    importantly, share the same 15-minute departure slot) act as a congestion proxy. These are target-free, so
    they transfer across the 2005 -> 2006 time shift, unlike target encodings and route identity (both measured
    to hurt here).
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


def _hour_of(df: pd.DataFrame) -> pd.Series:
    """Scheduled hour from the hhmm DepTime integer (2400-style values wrapped to 0)."""
    t = pd.to_numeric(df[DEPTIME], errors="coerce")
    t = t.where(t < 2400, t - 2400)
    h = np.floor(t / 100.0)
    return h.where((h >= 0) & (h <= 23), np.nan)


def _minofday(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df[DEPTIME], errors="coerce")
    t = t.where(t < 2400, t - 2400)
    h = np.floor(t / 100.0)
    m = t - h * 100.0
    mod = h * 60.0 + m
    return mod.where((h >= 0) & (h <= 23) & (m >= 0) & (m <= 59), np.nan)


def _str(series: pd.Series) -> pd.Series:
    return series.astype(str)


def _bucket15(df: pd.DataFrame) -> pd.Series:
    return (_minofday(df) // 15).astype("Int64").astype(str)


# --- timetable-density keys: log1p(count in training) per key -------------------
HAS_NETWORK = all(c in RAW_FEATURES for c in ("Origin", "Dest", "UniqueCarrier"))
DENS_SPECS = {
    "route": lambda df: _str(df["Origin"]) + "_" + _str(df["Dest"]),
    "origin": lambda df: _str(df["Origin"]),
    "dest": lambda df: _str(df["Dest"]),
    "carrier": lambda df: _str(df["UniqueCarrier"]),
    "origin_hour": lambda df: _str(df["Origin"]) + "@" + _hour_of(df).astype("Int64").astype(str),
    "dest_hour": lambda df: _str(df["Dest"]) + "@" + _hour_of(df).astype("Int64").astype(str),
    "carrier_hour": lambda df: _str(df["UniqueCarrier"]) + "@" + _hour_of(df).astype("Int64").astype(str),
    "route_hour": lambda df: (_str(df["Origin"]) + "_" + _str(df["Dest"]) + "@"
                              + _hour_of(df).astype("Int64").astype(str)),
    "origin_min15": lambda df: _str(df["Origin"]) + "#" + _bucket15(df),
    "dest_min15": lambda df: _str(df["Dest"]) + "#" + _bucket15(df),
    "route_min15": lambda df: (_str(df["Origin"]) + "_" + _str(df["Dest"]) + "#" + _bucket15(df)),
}
if HAS_NETWORK:
    DENS = {k: {kk: float(np.log1p(vv)) for kk, vv in f(train).value_counts().items()}
            for k, f in DENS_SPECS.items()}
else:
    DENS = {}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[RAW_FEATURES].copy()
    for c in ORD_COLS:
        X[c] = _ord_num(X[c])
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    if DEPTIME is not None:
        hour = _hour_of(df)
        mod = _minofday(df)
        X["dep_hour"] = hour
        X["dep_minute"] = mod - hour * 60.0
        X["dep_minofday"] = mod
    for name, spec in DENS_SPECS.items():
        if name in DENS:
            X[name + "_cnt"] = spec(df).map(DENS[name]).fillna(0.0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: small ensemble of strongly regularized lossguide boosters -----------
PARAMS = dict(
    n_estimators=500,
    learning_rate=0.02,
    grow_policy="lossguide",
    max_depth=0,
    max_leaves=256,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.5,
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
