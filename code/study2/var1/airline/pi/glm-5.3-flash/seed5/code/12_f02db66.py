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
    ("te_dest_hour", ["Dest", "hour"], 150),
    ("te_origin_dow", ["Origin", "dayofweek"], 100),
]
# congestion-volume features: (name, key columns) counted on train
VOL_SPECS = [
    ("vol_oh", ["Origin", "hour"]),
    ("vol_dh", ["Dest", "hour"]),
    ("vol_ow", ["Origin", "dayofweek"]),
    ("vol_rt", ["Origin", "Dest"]),
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
# congestion volumes counted on train (used by prepare); keyed by the same string keys as TE
vol_maps = {name: _te_key(_train_bf, cols).value_counts() for name, cols in VOL_SPECS}
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
    for name, cols in VOL_SPECS:
        X[name] = _te_key(X, cols).map(vol_maps[name]).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# small ensemble: diverse tree shapes + seeds, probability-averaged
CONFIGS = [
    dict(seed=42, colsample_bytree=0.8, n_estimators=350, max_depth=6),
    dict(seed=7, colsample_bytree=0.8, n_estimators=350, max_depth=0, grow_policy="lossguide", max_leaves=128),
    dict(seed=2024, colsample_bytree=0.9, n_estimators=500, max_depth=4),
    dict(seed=99, colsample_bytree=0.7, n_estimators=500, max_depth=0, grow_policy="lossguide", max_leaves=64),
    dict(seed=555, colsample_bytree=0.85, n_estimators=400, max_depth=5, subsample=0.7),
    dict(seed=31, colsample_bytree=0.75, n_estimators=400, max_depth=7),
    dict(seed=123, colsample_bytree=0.85, n_estimators=400, max_depth=0, grow_policy="lossguide", max_leaves=192),
    dict(seed=777, colsample_bytree=0.9, n_estimators=700, max_depth=3),
    dict(seed=55, colsample_bytree=0.8, n_estimators=400, max_depth=0, grow_policy="lossguide", max_leaves=96),
    dict(seed=321, colsample_bytree=0.7, n_estimators=400, max_depth=8),
]
# feature-subset members for diversity
TE_ONLY = [name for name, _cols, _m in TE_SPECS]
BASE_ONLY = ["hour", "minute", "month", "dayofmonth", "dayofweek", "distance"] + CAT_COLS + [name for name, _c in VOL_SPECS]
SUBSET_CONFIGS = [
    (dict(seed=31, colsample_bytree=0.8, n_estimators=350, max_depth=6), TE_ONLY),
    (dict(seed=77, colsample_bytree=0.8, n_estimators=350, max_depth=6), BASE_ONLY),
]


def make_model(cfg: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        max_depth=cfg["max_depth"],
        n_estimators=cfg["n_estimators"],
        learning_rate=0.03,
        subsample=cfg.get("subsample", 0.85),
        colsample_bytree=cfg["colsample_bytree"],
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=cfg["seed"],
        n_jobs=N_JOBS,
        **{k: v for k, v in cfg.items() if k in ("grow_policy", "max_leaves")},
    )


t0 = time.time()
X_train = prepare(train)
for name in oof_te.columns:  # replace map-based TE with OOF values to avoid leakage
    X_train[name] = oof_te[name].to_numpy()
y_train = to_y(train)
models = []
for cfg in CONFIGS:
    m = make_model(cfg)
    m.fit(X_train, y_train)
    models.append((m, None))
for cfg, cols in SUBSET_CONFIGS:
    m = make_model(cfg)
    m.fit(X_train[cols], y_train)
    models.append((m, cols))
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X if cols is None else X[cols])[:, 1] for m, cols in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
