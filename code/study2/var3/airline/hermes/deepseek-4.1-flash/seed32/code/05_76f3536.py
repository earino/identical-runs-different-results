"""PROBE (temporary): windowed + cumulative airport congestion features."""
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
Y_TR = (train[TARGET] == POSITIVE).astype(int).to_numpy()
Y_EV = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

RAW_FEATURES = [c for c in train.columns if c not in ID_COLS + [TARGET]]
_C_TOKEN = re.compile(r"^c-(\d+)$")


def _is_str(s):
    return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)


def _is_c_token(s):
    v = s.dropna().unique()
    return len(v) > 0 and all(_C_TOKEN.match(str(x)) for x in v)


def _ord_num(s):
    return pd.to_numeric(s.astype(str).str.replace("^c-", "", regex=True), errors="coerce")


ORD_COLS = [c for c in RAW_FEATURES if _is_str(train[c]) and _is_c_token(train[c])]
CAT_COLS = [c for c in RAW_FEATURES if _is_str(train[c]) and c not in ORD_COLS and train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _t(df):
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    return t.where(t < 2400, t - 2400)


def hour_of(df):
    h = np.floor(_t(df) / 100.0)
    return h.where((h >= 0) & (h <= 23), np.nan)


def minofday(df):
    t = _t(df)
    h = np.floor(t / 100.0)
    m = t - h * 100.0
    return (h * 60.0 + m).where((h >= 0) & (h <= 23) & (m >= 0) & (m <= 59), np.nan)


def b15(df):
    return (minofday(df) // 15).astype("Int64")


def o(df):
    return df["Origin"].astype(str)


def d(df):
    return df["Dest"].astype(str)


def c_(df):
    return df["UniqueCarrier"].astype(str)


def rt(df):
    return o(df) + "_" + d(df)


DENS_SPECS = {
    "route": rt, "origin": o, "dest": d, "carrier": c_,
    "origin_hour": lambda df: o(df) + "@" + hour_of(df).astype("Int64").astype(str),
    "dest_hour": lambda df: d(df) + "@" + hour_of(df).astype("Int64").astype(str),
    "carrier_hour": lambda df: c_(df) + "@" + hour_of(df).astype("Int64").astype(str),
    "route_hour": lambda df: rt(df) + "@" + hour_of(df).astype("Int64").astype(str),
    "origin_min15": lambda df: o(df) + "#" + b15(df).astype(str),
    "dest_min15": lambda df: d(df) + "#" + b15(df).astype(str),
    "route_min15": lambda df: rt(df) + "#" + b15(df).astype(str),
}
DENS = {k: {kk: float(np.log1p(vv)) for kk, vv in f(train).value_counts().items()} for k, f in DENS_SPECS.items()}
BASE_SET = ["route", "origin", "dest", "carrier", "origin_hour", "dest_hour", "carrier_hour", "route_hour",
            "origin_min15", "dest_min15", "route_min15"]

# --- windowed / cumulative congestion tables built from TRAIN ----------------
_tb = pd.DataFrame({"e": o(train), "b": b15(train).astype("int64")}).value_counts().reset_index(name="n")
_piv = _tb.pivot_table(index="e", columns="b", values="n", fill_value=0)
_piv = _piv.reindex(columns=range(96), fill_value=0)


def _table_lookup(piv, entity, bucket):
    """Map rows of a frame to values in a (entity x bucket) table."""
    idx = pd.MultiIndex.from_arrays([entity, bucket.astype("int64")])
    return piv.stack().reindex(idx).to_numpy()


WIN45 = _piv.rolling(window=3, min_periods=1, center=True, axis=1).sum()
WIN75 = _piv.rolling(window=5, min_periods=1, center=True, axis=1).sum()
_day_total = _piv.sum(axis=1)
CUMFRAC = _piv.cumsum(axis=1).div(_day_total, axis=0)
_piv_rt = pd.DataFrame({"e": rt(train), "b": b15(train).astype("int64")}).value_counts().reset_index(name="n")
_piv_rt = _piv_rt.pivot_table(index="e", columns="b", values="n", fill_value=0).reindex(columns=range(96), fill_value=0)
WIN45_RT = _piv_rt.rolling(window=3, min_periods=1, center=True, axis=1).sum()


def base(df):
    X = df[RAW_FEATURES].copy()
    for c in ORD_COLS:
        X[c] = _ord_num(X[c])
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    h = hour_of(df)
    m = minofday(df)
    X["dep_hour"] = h
    X["dep_minute"] = m - h * 60.0
    X["dep_minofday"] = m
    return X


def make_fs(extra=()):
    def fs(df):
        X = base(df)
        for n in BASE_SET:
            X[n + "_cnt"] = DENS_SPECS[n](df).map(DENS[n]).fillna(0.0).to_numpy()
        b = b15(df)
        if "win45" in extra:
            X["origin_win45"] = _table_lookup(WIN45, o(df), b)
            X["dest_win45"] = _table_lookup(WIN45, d(df), b)
        if "win75" in extra:
            X["origin_win75"] = _table_lookup(WIN75, o(df), b)
        if "win45rt" in extra:
            X["route_win45"] = _table_lookup(WIN45_RT, rt(df), b)
        if "cumfrac" in extra:
            X["origin_cumfrac"] = _table_lookup(CUMFRAC, o(df), b)
            X["dest_cumfrac"] = _table_lookup(CUMFRAC, d(df), b)
        if "ratios" in extra:
            X["origin_min15_vs_hour"] = X["origin_min15_cnt"] - X["origin_hour_cnt"]
            X["dest_min15_vs_hour"] = X["dest_min15_cnt"] - X["dest_hour_cnt"]
        return X
    return fs


CFG = dict(n_estimators=500, learning_rate=0.02, grow_policy="lossguide", max_depth=0, max_leaves=256,
           min_child_weight=50, subsample=0.8, colsample_bytree=0.5, reg_lambda=20.0, max_bin=512)


def run(name, fs):
    Xtr, Xev = fs(train), fs(evald)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **CFG)
    t0 = time.time()
    m.fit(Xtr, Y_TR)
    auc = roc_auc_score(Y_EV, m.predict_proba(Xev)[:, 1])
    print(f"PROBE {name:24s} auc={auc:.4f}  t={time.time() - t0:.1f}s  ncol={Xtr.shape[1]}", flush=True)


run("V0 ref", make_fs())
run("V1 +win45/win75", make_fs(("win45", "win75", "win45rt")))
run("V2 +cumfrac", make_fs(("cumfrac",)))
run("V3 +ratios", make_fs(("ratios",)))

Xtr, Xev = make_fs()(train), make_fs()(evald)
model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **CFG)
model.fit(Xtr, Y_TR)
print(f"Eval AUC: {roc_auc_score(Y_EV, model.predict_proba(Xev)[:, 1]):.4f}")
