"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()

# --- encoders/statistics fit on TRAINING DATA ONLY ------------------------------
CAT_LEVELS = {
    c: pd.Index(sorted(train[c].dropna().unique()))
    for c in ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
}
# target encoding: key -> Series of smoothed mean(y) ; computed on train, applied to any df
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "ROUTE", "HOUR", "ORIGIN_HOUR", "DEST_HOUR", "CARRIER_HOUR", "DTBLOCK"]
TE_SMOOTH = {"UniqueCarrier": 20, "Origin": 20, "Dest": 20, "ROUTE": 60, "HOUR": 30, "ORIGIN_HOUR": 60,
             "DEST_HOUR": 60, "CARRIER_HOUR": 40, "DTBLOCK": 40}
GLOBAL_MEAN = float(Y_TRAIN.mean())


def _te_key(df: pd.DataFrame, key: str) -> pd.Series:
    if key == "ROUTE":
        return df["Origin"].astype(str) + ">" + df["Dest"].astype(str)
    if key == "HOUR":
        return ((df["DepTime"] // 100) % 24).astype(str)
    if key == "DTBLOCK":
        return pd.cut(df["DepTime"], [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2700], right=False).astype(str)
    if key == "ORIGIN_HOUR":
        return df["Origin"].astype(str) + ">" + ((df["DepTime"] // 100) % 24).astype(str)
    if key == "DEST_HOUR":
        return df["Dest"].astype(str) + ">" + ((df["DepTime"] // 100) % 24).astype(str)
    if key == "CARRIER_HOUR":
        return df["UniqueCarrier"].astype(str) + ">" + ((df["DepTime"] // 100) % 24).astype(str)
    return df[key].astype(str)


def _te_stats(df: pd.DataFrame, y: np.ndarray, key: str, gm: float) -> dict:
    grp = pd.DataFrame({"k": _te_key(df, key), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    m = TE_SMOOTH[key]
    return ((grp["sum"] + m * gm) / (grp["count"] + m)).to_dict()


_TE_STATS = {k: _te_stats(train, Y_TRAIN, k, GLOBAL_MEAN) for k in TE_KEYS}


def _te_apply(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for k in TE_KEYS:
        out["te_" + k.lower()] = _te_key(df, k).map(_TE_STATS[k]).fillna(GLOBAL_MEAN).astype(float)
    return out


# out-of-fold target encodings for the TRAINING rows (avoid leakage into the fit)
FOLDS = 5
_kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
OOF_TE = pd.DataFrame(index=train.index, columns=["te_" + k.lower() for k in TE_KEYS], dtype=float)
for tr_idx, va_idx in _kf.split(train):
    sub = train.iloc[tr_idx]
    ysub = Y_TRAIN[tr_idx]
    gm = float(ysub.mean())
    stats = {k: _te_stats(sub, ysub, k, gm) for k in TE_KEYS}
    dva = train.iloc[va_idx]
    for k in TE_KEYS:
        OOF_TE.iloc[va_idx, OOF_TE.columns.get_loc("te_" + k.lower())] = (
            _te_key(dva, k).map(stats[k]).fillna(gm).to_numpy()
        )

# volume counts fit on train
_CNT_KEYS = [("carrier", "UniqueCarrier"), ("origin", "Origin"), ("dest", "Dest"), ("route", "ROUTE")]
_CNT_STATS = {name: _te_key(train, key).value_counts().astype(float) for name, key in _CNT_KEYS}


def _base_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Distance"] = df["Distance"].astype(float)
    hour = (df["DepTime"] // 100) % 24
    minute = df["DepTime"] % 100
    ang = 2 * np.pi * (hour * 60 + minute) / 1440
    X["dt_sin"] = np.sin(ang)
    X["dt_cos"] = np.cos(ang)
    X["DepTime"] = df["DepTime"].astype(float)
    month = df["Month"].str[2:].astype(int)
    dom = df["DayofMonth"].str[2:].astype(int)
    dow = df["DayOfWeek"].str[2:].astype(int)
    doy = (month * 31 + dom).astype(float)
    a2 = 2 * np.pi * doy / 372
    X["doy_sin"] = np.sin(a2)
    X["doy_cos"] = np.cos(a2)
    X["dow"] = dow.astype(float)
    X["is_night"] = ((hour >= 21) | (hour <= 4)).astype(float)
    # congestion: flights scheduled same day at the origin / destination (self-contained, no leakage)
    day_id = df["Month"].astype(str) + "-" + df["DayofMonth"].astype(str)
    _d = df.assign(_d=day_id)
    cong_o = _d.groupby(["_d", "Origin"], dropna=False)["Origin"].transform("size")
    cong_d = _d.groupby(["_d", "Dest"], dropna=False)["Dest"].transform("size")
    hb = pd.cut(df["DepTime"], [0, 600, 900, 1200, 1500, 1800, 2100, 2700], right=False).astype(str)
    _dh = df.assign(_d=day_id, _hb=hb)
    cong_ohb = _dh.groupby(["_d", "Origin", "_hb"], dropna=False)["Origin"].transform("size")
    cong_c = _d.groupby(["_d", "UniqueCarrier"], dropna=False)["UniqueCarrier"].transform("size")
    cong_day = _d.groupby("_d", dropna=False)["_d"].transform("size")
    X["cong_o"] = np.log1p(cong_o.astype(float))
    X["cong_d"] = np.log1p(cong_d.astype(float))
    X["cong_ohb"] = np.log1p(cong_ohb.astype(float))
    X["cong_c"] = np.log1p(cong_c.astype(float))
    X["cong_day"] = np.log1p(cong_day.astype(float))
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _base_features(df)
    X = pd.concat([X, _te_apply(df)], axis=1)
    for name, key in _CNT_KEYS:
        cnt = _te_key(df, key).map(_CNT_STATS[name]).fillna(0.0)
        X["cnt_" + name] = np.log1p(cnt)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model ---------------------------------------------------------------------
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 6,
    "min_child_weight": 50,
    "eta": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "seed": SEED,
    "nthread": N_JOBS,
}

X_ALL = prepare(train)
X_ALL[["te_" + k.lower() for k in TE_KEYS]] = OOF_TE.to_numpy()
Y_ALL = Y_TRAIN

X_EVAL = prepare(evald)
D_EVAL = xgb.DMatrix(X_EVAL, label=to_y(evald), enable_categorical=True)

# bagged ensemble: each seed trains on 90% of train, early-stops on the other 10%
N_BAGS = 6
# per-bag config diversity: (depth, colsample, min_child_weight)
BAG_CFGS = [
    (6, 0.7, 50),
    (8, 0.7, 50),
    (6, 0.5, 50),
    (8, 0.5, 50),
    (6, 0.9, 30),
    (8, 0.9, 30),
]
rng = np.random.RandomState(SEED)
t0 = time.time()
MODELS = []
for b in range(N_BAGS):
    idx = rng.permutation(len(X_ALL))
    va = idx[: len(idx) // 10]
    tr = idx[len(idx) // 10 :]
    dtr = xgb.DMatrix(X_ALL.iloc[tr], label=Y_ALL[tr], enable_categorical=True)
    dva = xgb.DMatrix(X_ALL.iloc[va], label=Y_ALL[va], enable_categorical=True)
    p = dict(PARAMS)
    p["seed"] = SEED + b
    p["max_depth"], p["colsample_bytree"], p["min_child_weight"] = BAG_CFGS[b]
    m = xgb.train(
        p,
        dtr,
        num_boost_round=3000,
        evals=[(dva, "va")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    MODELS.append(m)
    print(f"bag {b}: depth={p['max_depth']} col={p['colsample_bytree']} best_it={m.best_iteration} va_auc={m.best_score:.5f}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dm = xgb.DMatrix(prepare(df), enable_categorical=True)
    preds = [m.predict(dm, iteration_range=(0, m.best_iteration + 1)) for m in MODELS]
    return np.mean(preds, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
