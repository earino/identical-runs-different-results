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
TE_KEYS = ["UniqueCarrier", "Origin", "Dest", "ROUTE", "HOUR", "ORIGIN_HOUR"]
TE_SMOOTH = {"UniqueCarrier": 20, "Origin": 20, "Dest": 20, "ROUTE": 60, "HOUR": 30, "ORIGIN_HOUR": 60}
GLOBAL_MEAN = float(Y_TRAIN.mean())


def _te_key(df: pd.DataFrame, key: str) -> pd.Series:
    if key == "ROUTE":
        return df["Origin"].astype(str) + ">" + df["Dest"].astype(str)
    if key == "HOUR":
        return ((df["DepTime"] // 100) % 24).astype(str)
    if key == "ORIGIN_HOUR":
        return df["Origin"].astype(str) + ">" + ((df["DepTime"] // 100) % 24).astype(str)
    return df[key].astype(str)


_TE_STATS = {}
for k in TE_KEYS:
    grp = pd.DataFrame({"k": _te_key(train, k), "y": Y_TRAIN}).groupby("k")["y"].agg(["sum", "count"])
    m = TE_SMOOTH[k]
    _TE_STATS[k] = ((grp["sum"] + m * GLOBAL_MEAN) / (grp["count"] + m)).to_dict()


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
    stats = {}
    for k in TE_KEYS:
        grp = pd.DataFrame({"k": _te_key(sub, k), "y": ysub}).groupby("k")["y"].agg(["sum", "count"])
        m = TE_SMOOTH[k]
        stats[k] = ((grp["sum"] + m * gm) / (grp["count"] + m)).to_dict()
    dva = train.iloc[va_idx]
    for k in TE_KEYS:
        OOF_TE.iloc[va_idx, OOF_TE.columns.get_loc("te_" + k.lower())] = (
            _te_key(dva, k).map(stats[k]).fillna(gm).to_numpy()
        )

# volume counts fit on train
_CNT_STATS = {}
for name, key in [("carrier", "UniqueCarrier"), ("origin", "Origin"), ("dest", "Dest"), ("route", "ROUTE")]:
    _CNT_STATS[name] = _te_key(train, key).value_counts().astype(float)


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
    for c, levels in CAT_LEVELS.items():
        X[c] = pd.Categorical(df[c], categories=levels)
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _base_features(df)
    X = pd.concat([X, _te_apply(df)], axis=1)
    for name, key in [("carrier", "UniqueCarrier"), ("origin", "Origin"), ("dest", "Dest"), ("route", "ROUTE")]:
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
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
    "nthread": N_JOBS,
}

X_EVAL = prepare(evald)
DVAL = xgb.DMatrix(X_EVAL, label=to_y(evald), enable_categorical=True)

# training matrix: base+counts from prepare(), but TE columns replaced by OOF versions
X_TRAIN = prepare(train)
X_TRAIN[OOF_TE.columns] = OOF_TE.to_numpy()
DTRAIN = xgb.DMatrix(X_TRAIN, label=Y_TRAIN, enable_categorical=True)

t0 = time.time()
model = xgb.train(
    PARAMS,
    DTRAIN,
    num_boost_round=2000,
    evals=[(DTRAIN, "train"), (DVAL, "eval")],
    early_stopping_rounds=100,
    verbose_eval=100,
)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dm = xgb.DMatrix(prepare(df), enable_categorical=True)
    return model.predict(dm, iteration_range=(0, model.best_iteration + 1))


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
