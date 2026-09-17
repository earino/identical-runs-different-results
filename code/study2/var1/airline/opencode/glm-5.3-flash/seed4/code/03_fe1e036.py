"""XGBoost ensemble on the airline dataset (see program.md).

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Layout: all statistics (categorical levels, smoothed target-encoding maps) are fitted on the
TRAINING data only. Training rows get out-of-fold target encodings; any dataframe passed to
prepare()/predict_proba() gets encodings from the train-fitted maps. The final model is an
average of three XGBoost classifiers.
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
TE_M = 100  # target-encoding smoothing
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
_rt_tr = train["Origin"] + "_" + train["Dest"]
_rt_ev = evald["Origin"] + "_" + evald["Dest"]
# (name, train values, other-frame values) — "other" values are only used via train-fitted maps
TE_SPEC = [
    ("Origin", train["Origin"], evald["Origin"]),
    ("Dest", train["Dest"], evald["Dest"]),
    ("UniqueCarrier", train["UniqueCarrier"], evald["UniqueCarrier"]),
    ("Route", _rt_tr, _rt_ev),
    ("hour", _hr_tr, _hr_ev),
    ("OrgHour", train["Origin"].astype(str) + "_" + _hr_tr.astype(str), evald["Origin"].astype(str) + "_" + _hr_ev.astype(str)),
    ("DstHour", train["Dest"].astype(str) + "_" + _hr_tr.astype(str), evald["Dest"].astype(str) + "_" + _hr_ev.astype(str)),
    ("RtHour", _rt_tr.astype(str) + "_" + _hr_tr.astype(str), _rt_ev.astype(str) + "_" + _hr_ev.astype(str)),
    ("CarHour", train["UniqueCarrier"].astype(str) + "_" + _hr_tr.astype(str), evald["UniqueCarrier"].astype(str) + "_" + _hr_ev.astype(str)),
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
    TE_MAPS[_name] = _te_fit(_tv, y_all)

# OOF target encodings for the training rows (anti-leakage)
TE_OOF = {}
for _name, _tv, _ in TE_SPEC:
    _out = np.zeros(len(train))
    for _tri, _vai in KF.split(train):
        _out[_vai] = _te_apply(_te_fit(_tv.iloc[_tri], y_all[_tri]), _tv.iloc[_vai])
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
    X["mo_sin"] = np.sin(2 * np.pi * (mo - 1) / 12)
    X["mo_cos"] = np.cos(2 * np.pi * (mo - 1) / 12)
    X["dw_sin"] = np.sin(2 * np.pi * (dw - 1) / 7)
    X["dw_cos"] = np.cos(2 * np.pi * (dw - 1) / 7)
    hr = (df["DepTime"] // 100).clip(0, 23)
    rt = df["Origin"] + "_" + df["Dest"]
    te_vals = {
        "Origin": df["Origin"],
        "Dest": df["Dest"],
        "UniqueCarrier": df["UniqueCarrier"],
        "Route": rt,
        "hour": hr,
        "OrgHour": df["Origin"].astype(str) + "_" + hr.astype(str),
        "DstHour": df["Dest"].astype(str) + "_" + hr.astype(str),
        "RtHour": rt.astype(str) + "_" + hr.astype(str),
        "CarHour": df["UniqueCarrier"].astype(str) + "_" + hr.astype(str),
        "Month": df["Month"],
        "Dow": df["DayOfWeek"],
    }
    for name, vals in te_vals.items():
        X[name + "_te"] = _te_apply(TE_MAPS[name], vals)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models --------------------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
RAW_COLS = CAT_COLS + ["DepTime", "Distance"]
# (name, params, n_estimators, feature subset: "te" = full matrix, "raw" = cats+DepTime+Distance)
MODELS = [
    ("te10_d4", dict(max_depth=4, learning_rate=0.05, random_state=SEED), 200, "te"),
    ("te10_d3", dict(max_depth=3, learning_rate=0.05, random_state=SEED), 400, "te"),
    ("raw_d3", dict(max_depth=3, learning_rate=0.1, random_state=SEED), 300, "raw"),
]

X_FULL = prepare(train)
for _name, _, _ in TE_SPEC:
    X_FULL[_name + "_te"] = TE_OOF[_name]
X_RAW = X_FULL[RAW_COLS]


def _fit(name, params, n, subset):
    t0 = time.time()
    model = xgb.XGBClassifier(**BASE, **params, n_estimators=n)
    model.fit(X_FULL if subset == "te" else X_RAW, y_all)
    print(f"fit {name}: {time.time() - t0:.1f}s")
    return model


t0 = time.time()
ENSEMBLE = [(name, _fit(name, params, n, subset), X_FULL if subset == "te" else X_RAW) for name, params, n, subset in MODELS]
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X if Xc is X_FULL else X[RAW_COLS])[:, 1] for _, m, Xc in ENSEMBLE]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
