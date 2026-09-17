"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- feature specification ----------------------------------------------------
# raw string columns used as categoricals (low cardinality, transfer across years)
CAT_RAW = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_RAW}


# --- traffic-frequency statistics, fit on training data only -------------------
def _hour(series: pd.Series) -> np.ndarray:
    dep = pd.to_numeric(series, errors="coerce").fillna(-1).astype(float)
    return np.clip(dep // 100, 0, 23).to_numpy()

_tr_hour = _hour(train["DepTime"])
orig_cnt = train["Origin"].value_counts()
dest_cnt = train["Dest"].value_counts()
route_cnt = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts()
oh_cnt = train.groupby([train["Origin"], _tr_hour]).size()
dh_cnt = train.groupby([train["Dest"], _tr_hour]).size()
_tr_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
rh_cnt = train.assign(_r=_tr_route).groupby(["_r", _tr_hour]).size()
ch_cnt = train.groupby([train["UniqueCarrier"], _tr_hour]).size()
_tr_min = pd.to_numeric(train["DepTime"], errors="coerce").fillna(-1).astype(float) % 100
_tr_min = np.clip(_tr_min, 0, 59)
_tr_bucket = (_tr_hour * 60 + _tr_min) // 30
ob_cnt = train.assign(_b=_tr_bucket).groupby([train["Origin"], "_b"]).size()
rb_cnt = train.assign(_r=_tr_route, _b=_tr_bucket).groupby(["_r", "_b"]).size()
db_cnt = train.assign(_b=_tr_bucket).groupby([train["Dest"], "_b"]).size()
cb_cnt = train.assign(_b=_tr_bucket).groupby([train["UniqueCarrier"], "_b"]).size()
# network-diversity statistics
route_nc = train.assign(_r=_tr_route, _c=train["UniqueCarrier"]).groupby("_r")["_c"].nunique()
orig_nr = train.assign(_o=train["Origin"], _r=_tr_route).groupby("_o")["_r"].nunique()
carr_nr = train.assign(_c=train["UniqueCarrier"], _r=_tr_route).groupby("_c")["_r"].nunique()
orig_nc = train.assign(_o=train["Origin"], _c=train["UniqueCarrier"]).groupby("_o")["_c"].nunique()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype(float)
    hour = np.clip(dep // 100, 0, 23)
    minute = np.clip(dep % 100, 0, 59)
    frac = hour + minute / 60.0
    X["DepTime_hour"] = hour
    X["DepTime_minute"] = minute
    X["DepTime_frac"] = frac
    X["tod_sin"] = np.sin(2 * np.pi * frac / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * frac / 24.0)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce").astype(float)

    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    h = hour.to_numpy()
    X["orig_cnt"] = origin.map(orig_cnt).fillna(0.0).to_numpy()
    X["dest_cnt"] = dest.map(dest_cnt).fillna(0.0).to_numpy()
    X["route_cnt"] = (origin + "_" + dest).map(route_cnt).fillna(0.0).to_numpy()
    X["oh_cnt"] = oh_cnt.reindex(pd.MultiIndex.from_arrays([origin, h])).fillna(0.0).to_numpy()
    X["dh_cnt"] = dh_cnt.reindex(pd.MultiIndex.from_arrays([dest, h])).fillna(0.0).to_numpy()
    route = origin + "_" + dest
    X["rh_cnt"] = rh_cnt.reindex(pd.MultiIndex.from_arrays([route, h])).fillna(0.0).to_numpy()
    X["ch_cnt"] = ch_cnt.reindex(pd.MultiIndex.from_arrays([df["UniqueCarrier"].astype(str), h])).fillna(0.0).to_numpy()
    b = ((hour * 60 + minute) // 30).to_numpy()
    X["ob_cnt"] = ob_cnt.reindex(pd.MultiIndex.from_arrays([origin, b])).fillna(0.0).to_numpy()
    X["rb_cnt"] = rb_cnt.reindex(pd.MultiIndex.from_arrays([route, b])).fillna(0.0).to_numpy()
    X["db_cnt"] = db_cnt.reindex(pd.MultiIndex.from_arrays([dest, b])).fillna(0.0).to_numpy()
    X["cb_cnt"] = cb_cnt.reindex(pd.MultiIndex.from_arrays([df["UniqueCarrier"].astype(str), b])).fillna(0.0).to_numpy()
    # congestion in nearby 30-min windows
    for delta, nm in ((-2, "prev2"), (-1, "prev"), (1, "next"), (2, "next2")):
        bb = b + delta
        valid = (bb >= 0) & (bb <= 47)
        bb = np.clip(bb, 0, 47)
        X["ob_" + nm] = np.where(valid, ob_cnt.reindex(pd.MultiIndex.from_arrays([origin, bb])).fillna(0.0).to_numpy(), 0.0)
        X["rb_" + nm] = np.where(valid, rb_cnt.reindex(pd.MultiIndex.from_arrays([route, bb])).fillna(0.0).to_numpy(), 0.0)
    for delta, nm in ((-1, "prev"), (1, "next")):
        bb = b + delta
        valid = (bb >= 0) & (bb <= 47)
        bb = np.clip(bb, 0, 47)
        X["db_" + nm] = np.where(valid, db_cnt.reindex(pd.MultiIndex.from_arrays([dest, bb])).fillna(0.0).to_numpy(), 0.0)
        X["cb_" + nm] = np.where(valid, cb_cnt.reindex(pd.MultiIndex.from_arrays([df["UniqueCarrier"].astype(str), bb])).fillna(0.0).to_numpy(), 0.0)
    # network diversity and congestion shares
    X["route_nc"] = route.map(route_nc).fillna(0.0).to_numpy()
    X["orig_nr"] = origin.map(orig_nr).fillna(0.0).to_numpy()
    X["carr_nr"] = df["UniqueCarrier"].astype(str).map(carr_nr).fillna(0.0).to_numpy()
    X["orig_nc"] = origin.map(orig_nc).fillna(0.0).to_numpy()
    X["rb_share"] = X["rb_cnt"] / X["ob_cnt"].clip(lower=1)
    X["oh_share"] = X["oh_cnt"] / X["orig_cnt"].clip(lower=1)
    X["db_share"] = X["db_cnt"] / X["dest_cnt"].clip(lower=1)

    def catnum(s):
        return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")

    month = catnum(df["Month"])
    day = catnum(df["DayofMonth"])
    dow = catnum(df["DayOfWeek"])
    doy = (month - 1) * 30.0 + day
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 360.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 360.0)
    X["is_weekend"] = (dow >= 6).astype(float)
    X["dep_bucket"] = (hour // 3).astype(float)
    for c in CAT_RAW:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


X_all = prepare(train)
y_all = to_y(train)

# Deep, strongly L1-regularised trees transfer best across the 2005->2006 year shift.
# Average several seeds: deterministic and worth ~+0.0005 AUC.
N_SEEDS = 5
PARAMS = dict(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.4,
    min_child_weight=5,
    reg_lambda=1.0,
    reg_alpha=3.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    n_jobs=N_JOBS,
)

t0 = time.time()
models = []
for s in range(N_SEEDS):
    m = xgb.XGBClassifier(random_state=SEED + s, **PARAMS)
    m.fit(X_all, y_all, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({N_SEEDS} seeds)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    p = np.zeros(len(X))
    for m in models:
        p += m.predict_proba(X)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
