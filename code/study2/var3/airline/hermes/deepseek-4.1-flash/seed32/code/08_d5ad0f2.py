"""PROBE (temporary): busy-day density, asymmetric windows, seasonality."""
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


def _clock(df):
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    t = t.where(t < 2400, t - 2400)
    h = np.floor(t / 100.0)
    m = t - h * 100.0
    ok = (h >= 0) & (h <= 23) & (m >= 0) & (m <= 59)
    return h.where((h >= 0) & (h <= 23), np.nan), (h * 60.0 + m).where(ok, np.nan)


_S = lambda s: s.astype(str)
_OR = lambda df: _S(df["Origin"])
_DE = lambda df: _S(df["Dest"])
_CA = lambda df: _S(df["UniqueCarrier"])
_RT = lambda df: _OR(df) + "_" + _DE(df)
_H = lambda df: _clock(df)[0].astype("Int64").astype(str)
_B15 = lambda df: (_clock(df)[1] // 15).astype("Int64")
_MD = lambda df: _ord_num(df["Month"]).astype("Int64").astype(str) + "-" + _ord_num(df["DayofMonth"]).astype("Int64").astype(str)

DENS_SPECS = {
    "route": _RT, "origin": _OR, "dest": _DE, "carrier": _CA,
    "origin_hour": lambda df: _OR(df) + "@" + _H(df),
    "dest_hour": lambda df: _DE(df) + "@" + _H(df),
    "carrier_hour": lambda df: _CA(df) + "@" + _H(df),
    "route_hour": lambda df: _RT(df) + "@" + _H(df),
    "origin_min15": lambda df: _OR(df) + "#" + _B15(df).astype(str),
    "dest_min15": lambda df: _DE(df) + "#" + _B15(df).astype(str),
    "route_min15": lambda df: _RT(df) + "#" + _B15(df).astype(str),
    "md": _MD,
    "md_hour": lambda df: _MD(df) + "@" + _H(df),
}
DENS = {k: {kk: float(np.log1p(vv)) for kk, vv in f(train).value_counts().items()} for k, f in DENS_SPECS.items()}
BASE_SET = ["route", "origin", "dest", "carrier", "origin_hour", "dest_hour", "carrier_hour", "route_hour",
            "origin_min15", "dest_min15", "route_min15"]


def _slot_table(entity_fn):
    tb = pd.DataFrame({"e": entity_fn(train), "b": _B15(train).astype("int64")})
    tb = tb.value_counts().reset_index(name="n")
    piv = tb.pivot_table(index="e", columns="b", values="n", fill_value=0)
    return piv.reindex(columns=range(96), fill_value=0)


def _lookup(piv, entity, bucket):
    idx = pd.MultiIndex.from_arrays([entity, bucket.astype("int64")])
    return piv.stack().reindex(idx).to_numpy()


_piv_o, _piv_d, _piv_r = _slot_table(_OR), _slot_table(_DE), _slot_table(_RT)
_roll = lambda p, w: p.rolling(window=w, min_periods=1, center=True, axis=1).sum()
WIN_O45, WIN_D45, WIN_O75, WIN_R45 = _roll(_piv_o, 3), _roll(_piv_d, 3), _roll(_piv_o, 5), _roll(_piv_r, 3)
PREV_O15, NEXT_O15 = _piv_o.shift(1, axis=1), _piv_o.shift(-1, axis=1)


def base(df):
    X = df[RAW_FEATURES].copy()
    for c in ORD_COLS:
        X[c] = _ord_num(X[c])
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    h, m = _clock(df)
    X["dep_hour"] = h
    X["dep_minute"] = m - h * 60.0
    X["dep_minofday"] = m
    return X


def make_fs(extra=()):
    def fs(df):
        X = base(df)
        for n in BASE_SET:
            X[n + "_cnt"] = DENS_SPECS[n](df).map(DENS[n]).fillna(0.0).to_numpy()
        b = _B15(df)
        X["origin_win45"] = np.nan_to_num(_lookup(WIN_O45, _OR(df), b), nan=0.0)
        X["dest_win45"] = np.nan_to_num(_lookup(WIN_D45, _DE(df), b), nan=0.0)
        X["origin_win75"] = np.nan_to_num(_lookup(WIN_O75, _OR(df), b), nan=0.0)
        X["route_win45"] = np.nan_to_num(_lookup(WIN_R45, _RT(df), b), nan=0.0)
        X["origin_min15_vs_hour"] = X["origin_min15_cnt"] - X["origin_hour_cnt"]
        X["dest_min15_vs_hour"] = X["dest_min15_cnt"] - X["dest_hour_cnt"]
        if "md" in extra:
            X["md_cnt"] = DENS_SPECS["md"](df).map(DENS["md"]).fillna(0.0).to_numpy()
            X["md_hour_cnt"] = DENS_SPECS["md_hour"](df).map(DENS["md_hour"]).fillna(0.0).to_numpy()
        if "prevnext" in extra:
            X["origin_prev15"] = np.nan_to_num(_lookup(PREV_O15, _OR(df), b), nan=0.0)
            X["origin_next15"] = np.nan_to_num(_lookup(NEXT_O15, _OR(df), b), nan=0.0)
        if "doy" in extra:
            X["day_of_year"] = (_ord_num(df["Month"]) - 1) * 30.44 + _ord_num(df["DayofMonth"])
            X["is_weekend"] = (_ord_num(df["DayOfWeek"]) >= 6).astype(float)
        return X
    return fs


CFG = dict(n_estimators=500, learning_rate=0.02, grow_policy="lossguide", max_depth=0, max_leaves=256,
           min_child_weight=50, subsample=0.8, colsample_bytree=0.35, reg_lambda=20.0, max_bin=512)


def run(name, fs):
    Xtr, Xev = fs(train), fs(evald)
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS, **CFG)
    t0 = time.time()
    m.fit(Xtr, Y_TR)
    auc = roc_auc_score(Y_EV, m.predict_proba(Xev)[:, 1])
    print(f"PROBE {name:22s} auc={auc:.4f}  t={time.time() - t0:.1f}s  ncol={Xtr.shape[1]}", flush=True)
    return auc


run("Z1 +md density", make_fs(("md",)))
run("Z2 +prev/next15", make_fs(("prevnext",)))
run("Z3 +doy/weekend", make_fs(("doy",)))
z0 = run("Z0 ref (contract)", make_fs())
print(f"Eval AUC: {z0:.4f}")
