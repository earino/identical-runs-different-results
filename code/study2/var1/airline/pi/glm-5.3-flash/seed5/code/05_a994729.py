"""Airline delay XGBoost. Feature engineering lives in prepare(); encoders fitted on train only."""
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

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
# (name, key columns, smoothing m). Keys built from base_features columns.
TE_SPECS = [
    ("te_carrier", ["UniqueCarrier"], 30),
    ("te_origin", ["Origin"], 60),
    ("te_dest", ["Dest"], 60),
    ("te_carrier_hour", ["UniqueCarrier", "hour"], 150),
    ("te_origin_hour", ["Origin", "hour"], 150),
    ("te_route", ["Origin", "Dest"], 300),
    ("te_dow_hour", ["dayofweek", "hour"], 150),
]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def _int_from_c(s: pd.Series) -> pd.Series:
    return s.astype(str).str.slice(2).astype(float)


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise transforms only (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    X["hour"] = (df["DepTime"] // 100).astype(float)
    X["minute"] = (df["DepTime"] % 100).astype(float)
    X["month"] = _int_from_c(df["Month"])
    X["dayofmonth"] = _int_from_c(df["DayofMonth"])
    X["dayofweek"] = _int_from_c(df["DayOfWeek"])
    X["distance"] = df["Distance"].astype(float)
    X["Origin"] = df["Origin"].astype(str)
    X["Dest"] = df["Dest"].astype(str)
    X["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    return X


# --- fitted-on-train statistics (module level; applied inside prepare) ---------
_train_bf = base_features(train)
cat_levels = {c: pd.Index(sorted(_train_bf[c].unique())) for c in CAT_COLS}
_train_y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
_prior = float(_train_y.mean())


def _te_key(bf: pd.DataFrame, cols: list) -> pd.Series:
    parts = []
    for c in cols:
        s = bf[c]
        parts.append(s.astype(int).astype(str) if s.dtype.kind == "f" else s.astype(str))
    return parts[0] if len(parts) == 1 else parts[0] + "|" + parts[1]


def _smoothed_map(key: pd.Series, y: np.ndarray, m: float) -> pd.Series:
    g = pd.DataFrame({"k": key, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + m * _prior) / (g["count"] + m))


# full-train maps for unseen data (used by prepare)
te_maps = {name: _smoothed_map(_te_key(_train_bf, cols), _train_y, m) for name, cols, m in TE_SPECS}
# out-of-fold TE values for the training matrix (avoids leakage into fit)
oof_te = pd.DataFrame(index=train.index, dtype=float)
_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name, cols, m in TE_SPECS:
    key = _te_key(_train_bf, cols)
    oof_te[name] = _prior
    for fit_i, val_i in _kf.split(train):
        enc = _smoothed_map(key.iloc[fit_i], _train_y[fit_i], m)
        oof_te.iloc[val_i, oof_te.columns.get_loc(name)] = key.iloc[val_i].map(enc).to_numpy()
    oof_te[name] = oof_te[name].fillna(_prior)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = base_features(df)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for name, cols, _m in TE_SPECS:
        X[name] = _te_key(X, cols).map(te_maps[name]).fillna(_prior).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X_train = prepare(train)
for name in oof_te.columns:  # replace map-based TE with OOF values to avoid leakage
    X_train[name] = oof_te[name].to_numpy()
model.fit(X_train, to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
