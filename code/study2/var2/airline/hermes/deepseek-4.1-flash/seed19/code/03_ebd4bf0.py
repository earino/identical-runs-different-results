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
]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=df.index)
    d["Hour"] = _tod(df) // 60
    for c in ["Origin", "Dest", "UniqueCarrier"]:
        d[c] = df[c].astype(str)
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
for _spec in SPECS:
    _vc = pd.Series(_key(_dtrain, _spec)).value_counts()
    CNT_MAPS[_spec] = np.log1p(_vc).to_dict()
    RATE_MAPS[_spec] = _fit_rate(_dtrain, _ytr, _spec)[0]
    del _vc

# out-of-fold target rates for the training rows (the inference path uses the full-train maps above)
_fold = np.random.RandomState(SEED).permutation(len(_dtrain)) % N_FOLDS
OOF_RATE = {spec: np.empty(len(_dtrain)) for spec in SPECS}
for _f in range(N_FOLDS):
    _m = _fold == _f
    for _spec in SPECS:
        _rate, _prior = _fit_rate(_dtrain[~_m], _ytr[~_m], _spec)
        OOF_RATE[_spec][_m] = pd.Series(_key(_dtrain[_m], _spec)).map(_rate).fillna(_prior).to_numpy()


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
    return X


X_train = _base(train)
for spec in SPECS:
    name = "_".join(spec)
    X_train[f"CNT_{name}"] = pd.Series(_key(_dtrain, spec)).map(CNT_MAPS[spec]).fillna(0.0).to_numpy()
    X_train[f"TE_{name}"] = OOF_RATE[spec]

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(X_train, _ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
