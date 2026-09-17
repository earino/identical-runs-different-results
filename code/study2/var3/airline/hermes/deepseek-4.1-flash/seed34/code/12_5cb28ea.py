"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
# baseline: treat string columns as categoricals, but drop very high-cardinality ones (names, free text, timestamps)
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- key-column frequency encodings (fit on TRAIN ONLY) -----------------------
def _key_series(df: pd.DataFrame, kind: str) -> pd.Series:
    if kind == "Carrier":
        return df["UniqueCarrier"].astype(str)
    if kind == "Origin":
        return df["Origin"].astype(str)
    if kind == "Dest":
        return df["Dest"].astype(str)
    if kind == "Route":
        return df["Origin"].astype(str) + ">" + df["Dest"].astype(str)
    if kind == "CarrierOrigin":
        return df["UniqueCarrier"].astype(str) + ">" + df["Origin"].astype(str)
    raise KeyError(kind)


KEYS = ["Carrier", "Origin", "Dest", "Route", "CarrierOrigin"]
FREQ_MAPS = {k: _key_series(train, k).value_counts() for k in KEYS}
# airport connectivity / hub-degree and route-share, learned from train only
DEG_ORIGIN_DEST = train.groupby("Origin")["Dest"].nunique()
DEG_ORIGIN_CARRIER = train.groupby("Origin")["UniqueCarrier"].nunique()
DEG_DEST_ORIGIN = train.groupby("Dest")["Origin"].nunique()
_rt = pd.DataFrame({
    "r": _key_series(train, "Route").to_numpy(),
    "o": train["Origin"].astype(str).to_numpy(),
})
ROUTE_SHARE = (_rt.groupby("r")["o"].size() / _rt.groupby("r")["o"].first().map(FREQ_MAPS["Origin"])).dropna()


def _agg_block(df: pd.DataFrame) -> dict:
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    return {
        "deg_origin_dest": origin.map(DEG_ORIGIN_DEST).fillna(0.0).to_numpy(dtype="float32"),
        "deg_origin_carrier": origin.map(DEG_ORIGIN_CARRIER).fillna(0.0).to_numpy(dtype="float32"),
        "deg_dest_origin": dest.map(DEG_DEST_ORIGIN).fillna(0.0).to_numpy(dtype="float32"),
        "route_share_origin": _key_series(df, "Route").map(ROUTE_SHARE).fillna(0.0).to_numpy(dtype="float32"),
        "origin_share_carrier": (_key_series(df, "CarrierOrigin").map(FREQ_MAPS["CarrierOrigin"])
                                 / df["UniqueCarrier"].astype(str).map(FREQ_MAPS["Carrier"])
                                 ).fillna(0.0).to_numpy(dtype="float32"),
    }


def _cal_block(df: pd.DataFrame) -> dict:
    """Seasonal position: day-of-year (1..366) plus its annual cyclical pair."""
    mon = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False),
                        errors="coerce").to_numpy(dtype="float64")
    dom = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False),
                        errors="coerce").to_numpy(dtype="float64")
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype="float64")
    idx = np.clip(np.nan_to_num(mon, nan=1.0), 1, 12).astype(int) - 1
    doy = cum[idx] + dom
    return {
        "doy": doy,
        "doy_sin": np.sin(2.0 * np.pi * doy / 365.0),
        "doy_cos": np.cos(2.0 * np.pi * doy / 365.0),
        "doy2_sin": np.sin(4.0 * np.pi * doy / 365.0),
        "doy2_cos": np.cos(4.0 * np.pi * doy / 365.0),
        "woy": np.ceil(doy / 7.0),
    }


def _cat_labels(codes: np.ndarray, n: int) -> pd.Categorical:
    """Categorical with string labels from integer codes (anything outside [0, n) -> '__UNK__')."""
    labels = [str(i) for i in range(n)] + ["__UNK__"]
    idx = np.where((codes >= 0) & (codes < n), codes.astype("int64"), n)
    return pd.Categorical(np.asarray(labels)[idx], categories=labels)


def _hour_code(df: pd.DataFrame) -> np.ndarray:
    dt = pd.to_numeric(df["DepTime"], errors="coerce").to_numpy(dtype="float64")
    return np.rint(np.nan_to_num(np.mod(np.floor(dt / 100.0), 24.0), nan=-1.0)).astype("int64")


def _inter_key(df: pd.DataFrame, kind: str) -> pd.Series:
    """String keys for explicit categorical interactions."""
    hr = pd.Series(_hour_code(df), index=df.index).astype(str)
    if kind == "dow_hour":
        return df["DayOfWeek"].astype(str) + ":" + hr
    if kind == "carrier_hour":
        return df["UniqueCarrier"].astype(str) + ":" + hr
    if kind == "origin_hour":
        return df["Origin"].astype(str) + ":" + hr
    if kind == "dest_hour":
        return df["Dest"].astype(str) + ":" + hr
    raise KeyError(kind)


INTER_KINDS = ["dow_hour", "carrier_hour", "origin_hour", "dest_hour"]
INTER_LEVELS = {k: pd.Index(sorted(_inter_key(train, k).unique())) for k in INTER_KINDS}
INTER_FREQ = {k: _inter_key(train, k).value_counts() for k in INTER_KINDS}


def _time_block(df: pd.DataFrame) -> dict:
    """Scheduled departure time features (DepTime is hhmm; hh may be >= 24 for after-midnight flights)."""
    dt = pd.to_numeric(df["DepTime"], errors="coerce").to_numpy(dtype="float64")
    hh = np.floor(dt / 100.0)
    hc_i = np.rint(np.nan_to_num(np.mod(hh, 24.0), nan=-1.0)).astype("int64")
    tb_i = np.rint(np.nan_to_num(np.floor(dt / 200.0), nan=-1.0)).astype("int64")
    return {
        "hour": np.mod(hh, 24.0),
        "minute": dt - hh * 100.0,
        "log_dist": np.log1p(pd.to_numeric(df["Distance"], errors="coerce").to_numpy(dtype="float64")),
        "hour_cat": _cat_labels(hc_i, 24),
        "tod_bin": _cat_labels(tb_i, 14),
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for k in KEYS:
        X["freq_" + k] = _key_series(df, k).map(FREQ_MAPS[k]).fillna(0.0).astype("float32")
    for name, vals in _agg_block(df).items():
        X[name] = vals
    for name, vals in _cal_block(df).items():
        X[name] = vals
    for name, vals in _time_block(df).items():
        X[name] = vals
    for k in INTER_KINDS:
        key = _inter_key(df, k)
        X[k] = pd.Categorical(key, categories=INTER_LEVELS[k])
        X["freq_" + k] = key.map(INTER_FREQ[k]).fillna(0.0).astype("float32")
    return X


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    min_child_weight=30,
    subsample=0.7,
    colsample_bytree=0.6,
    reg_lambda=10.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 7, 2024, 31337]

t0 = time.time()
models = []
for s in SEEDS:
    m = xgb.XGBClassifier(**PARAMS, random_state=s)
    m.fit(prepare(train), to_y(train))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
