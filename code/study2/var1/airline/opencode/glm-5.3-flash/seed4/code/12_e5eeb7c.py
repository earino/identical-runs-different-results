"""XGBoost ensemble on the airline dataset (see program.md).

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Layout: all statistics (categorical levels, smoothed target-encoding maps) are fitted on the
TRAINING data only. Training rows get out-of-fold target encodings; any dataframe passed to
prepare()/predict_proba() gets encodings from the train-fitted maps. The final model is an
average of six XGBoost models (5 tree models + 1 linear gblinear model).
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
TE_M = 100  # default target-encoding smoothing
# per-feature smoothing: small m for low-cardinality columns (many rows per cell),
# larger m for high-cardinality interaction columns (sparse cells)
TE_M_BY_FEATURE = {
    "hour": 20, "Origin": 30, "Dest": 30, "UniqueCarrier": 30, "Month": 30, "Dow": 30,
    "OrgHour": 150, "DstHour": 150, "RtHour": 150, "CarHour": 150,
}
KF = KFold(n_splits=5, shuffle=True, random_state=SEED)

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
GLOBAL = y_all.mean()

# --- statistics fitted on TRAIN only ------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

_hr_tr = (train["DepTime"] // 100).clip(0, 23)
_hr_ev = (evald["DepTime"] // 100).clip(0, 23)
# half-hour buckets for the interaction encodings (finer time resolution)
_hh_tr = _hr_tr * 2 + (train["DepTime"] % 100 >= 30).astype(int)
_hh_ev = _hr_ev * 2 + (evald["DepTime"] % 100 >= 30).astype(int)
_rt_tr = train["Origin"] + "_" + train["Dest"]
_rt_ev = evald["Origin"] + "_" + evald["Dest"]
TE_SPEC = [
    ("Origin", train["Origin"], evald["Origin"]),
    ("Dest", train["Dest"], evald["Dest"]),
    ("UniqueCarrier", train["UniqueCarrier"], evald["UniqueCarrier"]),
    ("Route", _rt_tr, _rt_ev),
    ("hour", _hr_tr, _hr_ev),
    ("OrgHour", train["Origin"].astype(str) + "_" + _hh_tr.astype(str), evald["Origin"].astype(str) + "_" + _hh_ev.astype(str)),
    ("DstHour", train["Dest"].astype(str) + "_" + _hh_tr.astype(str), evald["Dest"].astype(str) + "_" + _hh_ev.astype(str)),
    ("RtHour", _rt_tr.astype(str) + "_" + _hh_tr.astype(str), _rt_ev.astype(str) + "_" + _hh_ev.astype(str)),
    ("CarHour", train["UniqueCarrier"].astype(str) + "_" + _hh_tr.astype(str), evald["UniqueCarrier"].astype(str) + "_" + _hh_ev.astype(str)),
    ("Month", train["Month"], evald["Month"]),
    ("Dow", train["DayOfWeek"], evald["DayOfWeek"]),
]
TE_MAPS = {}


def _te_fit(vals: pd.Series, yy: np.ndarray, m: int = TE_M) -> pd.Series:
    g = pd.DataFrame({"v": vals.to_numpy(), "_y": yy}).groupby("v")["_y"].agg(["mean", "count"])
    return (g["mean"] * g["count"] + GLOBAL * m) / (g["count"] + m)


def _te_apply(enc: pd.Series, vals: pd.Series) -> np.ndarray:
    return vals.map(enc).astype(float).fillna(GLOBAL).to_numpy()


for _name, _tv, _ev in TE_SPEC:
    TE_MAPS[_name] = _te_fit(_tv, y_all, TE_M_BY_FEATURE.get(_name, TE_M))

# OOF target encodings for the training rows (anti-leakage)
TE_OOF = {}
for _name, _tv, _ in TE_SPEC:
    _out = np.zeros(len(train))
    for _tri, _vai in KF.split(train):
        _out[_vai] = _te_apply(_te_fit(_tv.iloc[_tri], y_all[_tri], TE_M_BY_FEATURE.get(_name, TE_M)), _tv.iloc[_vai])
    TE_OOF[_name] = _out


# --- feature engineering -------------------------------------------------------
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    X["DepTime"] = df["DepTime"].astype("float64")
    X["Distance"] = df["Distance"].astype("float64")
    dt = df["DepTime"].astype("int64")
    tod = (((dt // 100) % 24) * 60 + dt % 100).to_numpy(dtype=float)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    mo = df["Month"].str.extract(r"c-(\d+)")[0].astype(float)
    dw = df["DayOfWeek"].str.extract(r"c-(\d+)")[0].astype(float)
    dm = df["DayofMonth"].str.extract(r"c-(\d+)")[0].astype(float)
    X["mo_sin"] = np.sin(2 * np.pi * (mo - 1) / 12)
    X["mo_cos"] = np.cos(2 * np.pi * (mo - 1) / 12)
    X["dw_sin"] = np.sin(2 * np.pi * (dw - 1) / 7)
    X["dw_cos"] = np.cos(2 * np.pi * (dw - 1) / 7)
    X["doy"] = ((mo - 1) * 30.4 + dm).astype(float)
    hr = (df["DepTime"] // 100).clip(0, 23)
    hh = hr * 2 + (df["DepTime"].astype("int64") % 100 >= 30).astype(int)
    rt = df["Origin"] + "_" + df["Dest"]
    te_vals = {
        "Origin": df["Origin"],
        "Dest": df["Dest"],
        "UniqueCarrier": df["UniqueCarrier"],
        "Route": rt,
        "hour": hr,
        "OrgHour": df["Origin"].astype(str) + "_" + hh.astype(str),
        "DstHour": df["Dest"].astype(str) + "_" + hh.astype(str),
        "RtHour": rt.astype(str) + "_" + hh.astype(str),
        "CarHour": df["UniqueCarrier"].astype(str) + "_" + hh.astype(str),
        "Month": df["Month"],
        "Dow": df["DayOfWeek"],
    }
    for name, vals in te_vals.items():
        X[name + "_te"] = _te_apply(TE_MAPS[name], vals)
    X["xmas"] = ((mo == 12) & (dm >= 18)).astype(float) + ((mo == 1) & (dm <= 3)).astype(float)
    X["tg"] = ((mo == 11) & (dm >= 18) & (dm <= 27)).astype(float)
    X["jul4"] = ((mo == 7) & (dm <= 5)).astype(float)
    X["ld"] = ((mo == 9) & (dm <= 7)).astype(float)
    X["md"] = ((mo == 5) & (dm >= 25)).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models --------------------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
RAW_COLS = CAT_COLS + ["DepTime", "Distance", "doy"]
NUM_COLS = [c for c in prepare(evald).columns if c not in CAT_COLS]
# (name, params, n_estimators, feature subset)
MODELS = [
    ("te_d4", dict(max_depth=4, learning_rate=0.05, random_state=SEED), 200, "te"),
    ("te_d3", dict(max_depth=3, learning_rate=0.05, random_state=SEED), 400, "te"),
    ("te_d2", dict(max_depth=2, learning_rate=0.05, random_state=SEED), 700, "te"),
    ("gbl", dict(booster="gblinear", learning_rate=0.1, objective="binary:logistic"), 300, "num"),
    ("raw_d3", dict(max_depth=3, learning_rate=0.1, random_state=SEED), 300, "raw"),
    ("d3n", dict(max_depth=3, learning_rate=0.05, random_state=SEED), 400, "num"),
    ("d4n", dict(max_depth=4, learning_rate=0.05, random_state=SEED), 200, "num"),
]

X_FULL = prepare(train)
for _name, _, _ in TE_SPEC:
    X_FULL[_name + "_te"] = TE_OOF[_name]
X_RAW = X_FULL[RAW_COLS]
X_NUM = X_FULL[NUM_COLS]
X_SUB = X_FULL.drop(columns=["UniqueCarrier", "CarHour_te"])
SUBSETS = {"te": X_FULL, "raw": X_RAW, "num": X_NUM, "sub": X_SUB}
COLS = {"te": list(X_FULL.columns), "raw": RAW_COLS, "num": NUM_COLS, "sub": list(X_SUB.columns)}


def _fit(name, params, n, subset):
    t0 = time.time()
    model = xgb.XGBClassifier(**BASE, **params, n_estimators=n)
    model.fit(SUBSETS[subset], y_all)
    print(f"fit {name}: {time.time() - t0:.1f}s")
    return model


def _fit_seedavg(name, params, n, subset, seeds=(SEED, 7, 13, 21, 5)):
    t0 = time.time()
    models = []
    for s in seeds:
        m = xgb.XGBClassifier(**BASE, **{**params, "random_state": s}, n_estimators=n)
        m.fit(SUBSETS[subset], y_all)
        models.append(m)
    print(f"fit {name} (seeds {seeds}): {time.time() - t0:.1f}s")
    return models


t0 = time.time()
ENSEMBLE = []
for name, params, n, subset in MODELS:
    if name in ("te_d4", "te_d3", "te_d2", "raw_d3"):  # seed-average noisy members
        ENSEMBLE.append((name, _fit_seedavg(name, params, n, subset), subset))
    else:
        ENSEMBLE.append((name, _fit(name, params, n, subset), subset))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = []
    for _, m, s in ENSEMBLE:
        preds = [mm.predict_proba(X[COLS[s]])[:, 1] for mm in (m if isinstance(m, list) else [m])]
        ps.append(np.mean(preds, axis=0))
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
