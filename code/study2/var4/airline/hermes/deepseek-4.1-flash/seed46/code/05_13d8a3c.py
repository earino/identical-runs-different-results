"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Design notes:
  * All feature engineering lives in `prepare(df)` (the code path `predict_proba` uses). The single exception is
    `prepare_train()`, which calls `prepare()` and then substitutes out-of-fold target-encoding values for the
    training rows only -- trained on OOF encodings, everything else sees the full-train encoding tables.
  * Encoders/statistics are fit on data/train.csv only, never on the dataframe being predicted.
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
N_FOLDS = 5
SMOOTH = 20.0

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- base features ------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols
            if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _num_level(series: pd.Series) -> pd.Series:
    """'c-12' -> 12.0 (the calendar columns are stored as c-<n> strings)."""
    return pd.to_numeric(series.astype("string").str.replace("c-", "", regex=False), errors="coerce")


def _dep_hour(df: pd.DataFrame) -> pd.Series:
    """Scheduled departure hour. DepTime is hhmm but contains invalid values up to ~2620."""
    dep = pd.to_numeric(df["DepTime"], errors="coerce").astype("float64") % 2400.0
    return (dep // 100.0).clip(0, 23)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    """String keys used by the target encoders (computed identically for train and unseen rows)."""
    k = pd.DataFrame(index=df.index)
    origin = df["Origin"].astype("string")
    dest = df["Dest"].astype("string")
    carrier = df["UniqueCarrier"].astype("string")
    hour = _dep_hour(df).astype("int16").astype("string")
    month = _num_level(df["Month"]).astype("int16").astype("string")
    dow = _num_level(df["DayOfWeek"]).astype("int16").astype("string")
    route = origin + "_" + dest
    k["route"] = route
    k["origin"] = origin
    k["dest"] = dest
    k["carrier"] = carrier
    k["hour"] = hour
    k["month"] = month
    k["dow"] = dow
    k["carrier_hour"] = carrier + "_" + hour
    k["origin_hour"] = origin + "_" + hour
    k["dest_hour"] = dest + "_" + hour
    k["route_hour"] = route + "_" + hour
    k["month_hour"] = month + "_" + hour
    k["route_dow"] = route + "_" + dow
    k["origin_dow"] = origin + "_" + dow
    k["dest_dow"] = dest + "_" + dow
    k["carrier_dow"] = carrier + "_" + dow
    k["hour_dow"] = hour + "_" + dow
    return k


TE_SPECS = ["route", "origin", "dest", "carrier", "hour", "month", "dow",
            "carrier_hour", "origin_hour", "dest_hour", "route_hour", "month_hour",
            "route_dow", "origin_dow", "dest_dow", "carrier_dow", "hour_dow"]
TE_COLS = [f"te_{c}" for c in TE_SPECS] + [f"cnt_{c}" for c in TE_SPECS]


def _fit_tables(keys: pd.DataFrame, y: np.ndarray, prior: float):
    """Smoothed positive rate + log count per key value."""
    tables = {}
    for spec in TE_SPECS:
        g = pd.DataFrame({"k": keys[spec].to_numpy(), "y": y}).groupby("k", sort=False)["y"]
        agg = g.agg(["sum", "count"])
        te = (agg["sum"] + prior * SMOOTH) / (agg["count"] + SMOOTH)
        tables[spec] = (te, np.log1p(agg["count"]))
    return tables


TRAIN_KEYS = _keys(train)
Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(Y_TRAIN.mean())
FULL_TABLES = _fit_tables(TRAIN_KEYS, Y_TRAIN, PRIOR)

# --- out-of-fold encodings for the training rows ------------------------------
_oof = np.empty((len(train), len(TE_SPECS)), dtype="float32")
_oof_cnt = np.empty((len(train), len(TE_SPECS)), dtype="float32")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(TRAIN_KEYS):
    fold_tables = _fit_tables(TRAIN_KEYS.iloc[tr_idx], Y_TRAIN[tr_idx], float(Y_TRAIN[tr_idx].mean()))
    for j, spec in enumerate(TE_SPECS):
        te, cnt = fold_tables[spec]
        _oof[va_idx, j] = TRAIN_KEYS[spec].iloc[va_idx].map(te).to_numpy(dtype="float32")
        _oof_cnt[va_idx, j] = TRAIN_KEYS[spec].iloc[va_idx].map(cnt).to_numpy(dtype="float32")
OOF_TE = pd.DataFrame(np.hstack([_oof, _oof_cnt]), columns=TE_COLS)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN

    dep = pd.to_numeric(df["DepTime"], errors="coerce").astype("float64") % 2400.0
    hour = (dep // 100.0).clip(0, 23)
    minute = (dep % 100.0).clip(0, 59)
    tod = hour * 60.0 + minute
    X["dep_hour"] = hour
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    dow = _num_level(df["DayOfWeek"])
    X["dow_num"] = dow
    X["is_weekend"] = dow.isin([6.0, 7.0]).astype("int8")

    keys = _keys(df)
    for spec in TE_SPECS:
        te, cnt = FULL_TABLES[spec]
        X[f"te_{spec}"] = keys[spec].map(te).astype("float32")
        X[f"cnt_{spec}"] = keys[spec].map(cnt).astype("float32")
    return X


def prepare_train() -> pd.DataFrame:
    """Training matrix: `prepare(train)` with target encodings replaced by out-of-fold values."""
    X = prepare(train)
    X[TE_COLS] = OOF_TE.to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 202, 777]

X_train = prepare_train()
y_train = to_y(train)
models = []
t0 = time.time()
for s in SEEDS:
    m = xgb.XGBClassifier(**PARAMS, random_state=s)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s for {len(models)} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
