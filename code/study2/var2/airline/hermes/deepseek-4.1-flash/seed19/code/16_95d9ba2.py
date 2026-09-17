"""XGBoost binary classifier for the airline delay task (autoresearch benchmark).

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

All feature engineering lives in prepare()/_derive(), which predict_proba() calls on raw rows; every statistic
(group target rates, traffic counts) is fitted on data/train.csv only. Target rates are cross-fitted (5-fold
out-of-fold) for the training matrix so the model cannot over-trust in-sample group statistics.
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
SMOOTH = 30.0  # target-rate shrinkage towards the global rate
N_FOLDS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
DROP_COLS = ["Origin", "Dest", "UniqueCarrier", "DayofMonth"]  # encoded by the group-rate/count features instead
feature_cols = [c for c in feature_cols if c not in DROP_COLS]
cat_cols = [c for c in cat_cols if c in feature_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _tod(df: pd.DataFrame) -> np.ndarray:
    """Scheduled departure as minutes past midnight (hhmm integers; 2400+ wraps to 0)."""
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    return ((dt // 100) % 24) * 60 + np.minimum(dt % 100, 59)


# group keys for the traffic-volume and delay-rate features: interactions the trees cannot easily build
SPECS = [
    ("Origin",),
    ("Dest",),
    ("UniqueCarrier",),
    ("Origin", "Dest"),
    ("Origin", "Hour"),
    ("Dest", "Hour"),
    ("UniqueCarrier", "Hour"),
    ("Origin", "Dest", "Hour"),
    ("Origin", "Month"),
    ("UniqueCarrier", "Origin"),
    ("TodBin",),
    ("Origin", "TodBin"),
    ("Dest", "TodBin"),
    ("UniqueCarrier", "TodBin"),
    ("Tod15",),
    ("Origin", "Tod15"),
    ("Dest", "Tod15"),
    ("Tod10",),
    ("Tod5",),
    ("UniqueCarrier", "Tod15"),
]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["Hour"] = _tod(df) // 60
    d["TodBin"] = _tod(df) // 30
    d["Tod15"] = _tod(df) // 15
    d["Tod10"] = _tod(df) // 10
    d["Tod5"] = _tod(df) // 5
    for c in ["Origin", "Dest", "UniqueCarrier", "Month"]:
        d[c] = df[c].astype(str)
    # ordinal calendar position: lets the trees use seasonality/weekly shape with few splits
    d["MonthNum"] = pd.to_numeric(df["Month"].astype(str).str.slice(2), errors="coerce")
    d["DOWNum"] = pd.to_numeric(df["DayOfWeek"].astype(str).str.slice(2), errors="coerce")
    d["DOY"] = d["MonthNum"] * 30.4 + pd.to_numeric(df["DayofMonth"].astype(str).str.slice(2), errors="coerce")
    return d


def _key(d: pd.DataFrame, spec) -> np.ndarray:
    k = d[spec[0]].to_numpy().astype(str)
    for c in spec[1:]:
        k = np.char.add(np.char.add(k, "|"), d[c].to_numpy().astype(str))
    return k


def _fit_rate(d: pd.DataFrame, y: np.ndarray, spec, smooth: float = SMOOTH):
    """Smoothed P(delayed) per group key, fitted on the rows given (training rows only)."""
    prior = float(y.mean())
    g = pd.DataFrame({"k": _key(d, spec), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    rate = ((g["sum"] + prior * smooth) / (g["count"] + smooth)).to_dict()
    return rate, prior


_dtrain = _derive(train)
_ytr = to_y(train)
CNT_MAPS = {}
RATE_MAPS = {}
RAW_CNT = {}
for _spec in SPECS:
    _vc = pd.Series(_key(_dtrain, _spec)).value_counts()
    CNT_MAPS[_spec] = np.log1p(_vc).to_dict()
    RAW_CNT[_spec] = _vc.to_dict()
    RATE_MAPS[_spec] = _fit_rate(_dtrain, _ytr, _spec)[0]
    del _vc

# share of a parent group's traffic that falls in a sub-group (congestion/peakiness, scale-free across years)
RATIO_SPECS = [
    (("Origin", "Hour"), ("Origin",)),
    (("Dest", "Hour"), ("Dest",)),
    (("Origin", "Dest", "Hour"), ("Origin", "Dest")),
]

# out-of-fold target rates for the training rows (the inference path uses the full-train maps above)
_fold = np.random.RandomState(SEED).permutation(len(_dtrain)) % N_FOLDS
OOF_RATE = {spec: np.empty(len(_dtrain)) for spec in SPECS}
for _f in range(N_FOLDS):
    _m = _fold == _f
    for _spec in SPECS:
        _rate, _prior = _fit_rate(_dtrain[~_m], _ytr[~_m], _spec)
        OOF_RATE[_spec][_m] = pd.Series(_key(_dtrain[_m], _spec)).map(_rate).fillna(_prior).to_numpy()


def _cal_features(X: pd.DataFrame, d: pd.DataFrame) -> None:
    for c in ["MonthNum", "DOWNum", "DOY"]:
        X[c] = d[c].to_numpy()


def _base(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = _base(df)
    d = _derive(df)
    for spec in SPECS:
        name = "_".join(spec)
        k = _key(d, spec)
        sp = pd.Series(k)
        X[f"CNT_{name}"] = sp.map(CNT_MAPS[spec]).fillna(0.0).to_numpy()
        X[f"TE_{name}"] = sp.map(RATE_MAPS[spec]).fillna(0.5).to_numpy()
    for child, parent in RATIO_SPECS:
        c = pd.Series(_key(d, child)).map(RAW_CNT[child]).fillna(0.0).to_numpy()
        p = pd.Series(_key(d, parent)).map(RAW_CNT[parent]).fillna(0.0).to_numpy()
        X[f"SHARE_{'_'.join(child)}"] = np.where(p > 0, c / np.maximum(p, 1.0), 0.0)
    _cal_features(X, d)
    return X


X_train = _base(train)
for spec in SPECS:
    name = "_".join(spec)
    X_train[f"CNT_{name}"] = pd.Series(_key(_dtrain, spec)).map(CNT_MAPS[spec]).fillna(0.0).to_numpy()
    X_train[f"TE_{name}"] = OOF_RATE[spec]
for child, parent in RATIO_SPECS:
    c = pd.Series(_key(_dtrain, child)).map(RAW_CNT[child]).fillna(0.0).to_numpy()
    p = pd.Series(_key(_dtrain, parent)).map(RAW_CNT[parent]).fillna(0.0).to_numpy()
    X_train[f"SHARE_{'_'.join(child)}"] = np.where(p > 0, c / np.maximum(p, 1.0), 0.0)
_cal_features(X_train, _dtrain)

# --- model --------------------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
MEMBERS = [
    dict(n_estimators=1500, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0),
    dict(n_estimators=1500, max_depth=4, learning_rate=0.02, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0),
    dict(n_estimators=2000, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.9, min_child_weight=10, reg_lambda=2.0),
    dict(n_estimators=1000, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=40, reg_lambda=10.0),
    dict(n_estimators=1500, max_depth=4, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6, min_child_weight=30, reg_lambda=5.0),
]
SEEDS = [42, 7, 2024, 99, 1234]

t0 = time.time()
models = []
for _p, _s in zip(MEMBERS, SEEDS):
    _m = xgb.XGBClassifier(random_state=_s, **BASE, **_p)
    _m.fit(X_train, _ytr)
    models.append(_m)
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
