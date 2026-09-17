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
from sklearn.model_selection import KFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- raw schema ---------------------------------------------------------------
RAW_COLS = [c for c in train.columns if c not in ID_COLS + [TARGET]]
OBJ_COLS = [c for c in RAW_COLS if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in OBJ_COLS if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}
NUM_COLS = [c for c in RAW_COLS if c not in OBJ_COLS]
C_NUM = [c for c in cat_cols if train[c].str.startswith("c-", na=False).all()]  # Month / DayofMonth / DayOfWeek


def _dep_minutes(df: pd.DataFrame) -> np.ndarray:
    """Scheduled DepTime (hhmm integer) -> minutes since midnight; invalid values -> NaN."""
    t = pd.to_numeric(df["DepTime"], errors="coerce").to_numpy(dtype="float64")
    hh = np.floor(t / 100.0)
    mm = t - hh * 100.0
    return np.where((hh >= 0) & (hh <= 23) & (mm >= 0) & (mm <= 59), hh * 60.0 + mm, np.nan)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in NUM_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        if c in C_NUM:
            if c == "Month":
                X["month_sin"] = np.sin(2 * np.pi * _c_int(df, c) / 12.0)
                X["month_cos"] = np.cos(2 * np.pi * _c_int(df, c) / 12.0)
            elif c == "DayOfWeek":
                X["dow_sin"] = np.sin(2 * np.pi * _c_int(df, c) / 7.0)
                X["dow_cos"] = np.cos(2 * np.pi * _c_int(df, c) / 7.0)
        else:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = _dep_minutes(df)
    X["dep_min"] = dep
    X["dep_hour"] = dep / 60.0
    X["dep_hour_sin"] = np.sin(2 * np.pi * dep / 1440.0)
    X["dep_hour_cos"] = np.cos(2 * np.pi * dep / 1440.0)
    return X


def _c_int(df: pd.DataFrame, c: str) -> np.ndarray:
    return pd.to_numeric(df[c].astype(str).str.replace("c-", "", regex=False), errors="coerce").to_numpy(dtype="float64")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target / frequency encodings (fit on TRAIN keys and labels only) ----------
Y = to_y(train)
PRIOR = Y.mean()
TE_SPEC = {
    "UniqueCarrier": 20.0,
    "Origin": 20.0,
    "Dest": 20.0,
    "route": 10.0,
    "carrier_hour": 20.0,
    "origin_hour": 20.0,
    "dest_hour": 20.0,
    "carrier_route": 5.0,
    "route_hour": 15.0,
    "origin_dow": 20.0,
    "dest_dow": 20.0,
    "carrier_dow": 20.0,
    "dow_hour": 20.0,
    "carrier_origin": 10.0,
    "carrier_dest": 10.0,
    "carrier_route_hour": 5.0,
}
TE_TRAIN_COLS = list(TE_SPEC)
FREQ_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "carrier_hour", "origin_hour", "dest_hour",
             "route_hour", "dow_hour", "carrier_route_hour", "carrier_route", "carrier_dow", "origin_dow",
             "dest_dow", "carrier_origin", "carrier_dest"]
RATIOS = [
    ("origin_hour", "Origin"),
    ("dest_hour", "Dest"),
    ("carrier_hour", "UniqueCarrier"),
    ("route_hour", "route"),
    ("carrier_route_hour", "carrier_route"),
    ("carrier_route", "route"),
    ("origin_dow", "Origin"),
    ("carrier_dow", "UniqueCarrier"),
]


def _encode_keys(df: pd.DataFrame) -> pd.DataFrame:
    key = pd.DataFrame(index=df.index)
    key["UniqueCarrier"] = df["UniqueCarrier"].astype(str)
    key["Origin"] = df["Origin"].astype(str)
    key["Dest"] = df["Dest"].astype(str)
    key["route"] = key["Origin"] + "_" + key["Dest"]
    hour = np.floor(_dep_minutes(df) / 60.0)
    hs = pd.Series(np.nan_to_num(hour, nan=-1).astype(int).astype(str), index=df.index)
    key["carrier_hour"] = key["UniqueCarrier"] + "_" + hs
    key["origin_hour"] = key["Origin"] + "_" + hs
    key["dest_hour"] = key["Dest"] + "_" + hs
    key["carrier_route"] = key["UniqueCarrier"] + "_" + key["route"]
    dow = _c_int(df, "DayOfWeek")
    ds = pd.Series(np.nan_to_num(dow, nan=-1).astype(int).astype(str), index=df.index)
    key["route_hour"] = key["route"] + "_" + hs
    key["origin_dow"] = key["Origin"] + "_" + ds
    key["dest_dow"] = key["Dest"] + "_" + ds
    key["carrier_dow"] = key["UniqueCarrier"] + "_" + ds
    key["dow_hour"] = ds + "_" + hs
    key["carrier_origin"] = key["UniqueCarrier"] + "_" + key["Origin"]
    key["carrier_dest"] = key["UniqueCarrier"] + "_" + key["Dest"]
    key["carrier_route_hour"] = key["carrier_route"] + "_" + hs
    return key


def _smooth_map(keys: pd.Series, y: np.ndarray, k: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + PRIOR * k) / (g["count"] + k)


_keys = _encode_keys(train)
TE_MAP = {c: _smooth_map(_keys[c], Y, TE_SPEC[c]) for c in TE_TRAIN_COLS}
FREQ_MAP = {c: _keys[c].value_counts() for c in FREQ_COLS}

# out-of-fold encodings for the training rows, so the model never sees its own labels
OOF = pd.DataFrame(index=train.index, columns=TE_TRAIN_COLS, dtype="float64")
for tr_i, va_i in KFold(n_splits=10, shuffle=True, random_state=SEED).split(train):
    for c in TE_TRAIN_COLS:
        OOF.loc[OOF.index[va_i], c] = _keys[c].iloc[va_i].map(_smooth_map(_keys[c].iloc[tr_i], Y[tr_i], TE_SPEC[c])).to_numpy()


def _add_te(X: pd.DataFrame, key: pd.DataFrame, oof: bool = False) -> pd.DataFrame:
    for c in TE_TRAIN_COLS:
        if oof:
            X[f"te_{c}"] = OOF[c].reindex(X.index).to_numpy()
        else:
            X[f"te_{c}"] = key[c].map(TE_MAP[c]).to_numpy(dtype="float64")
    for c in FREQ_COLS:
        X[f"freq_{c}"] = key[c].map(FREQ_MAP[c]).to_numpy(dtype="float64")
    # relative traffic shares: congestion intensity rather than airport size (year-stable)
    for num, den in RATIOS:
        d = X[f"freq_{den}"]
        X[f"share_{num}"] = X[f"freq_{num}"] / d.replace(0.0, np.nan)
    return X


# --- model --------------------------------------------------------------------
PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)
SEEDS = [42, 7]
CONFIGS = [
    dict(max_depth=3, colsample_bytree=0.8, learning_rate=0.10, n_estimators=150),
    dict(max_depth=4, colsample_bytree=0.5, learning_rate=0.10, n_estimators=150),
    dict(max_depth=4, colsample_bytree=0.8, learning_rate=0.08, n_estimators=180),
    dict(max_depth=5, colsample_bytree=0.5, learning_rate=0.10, n_estimators=180),
    dict(max_depth=5, colsample_bytree=0.7, learning_rate=0.08, n_estimators=150),
    dict(max_depth=5, colsample_bytree=0.9, learning_rate=0.05, n_estimators=250, min_child_weight=20),
    dict(max_depth=6, colsample_bytree=0.4, learning_rate=0.10, n_estimators=120),
    dict(max_depth=6, colsample_bytree=0.6, learning_rate=0.15, n_estimators=80),
    dict(max_depth=6, colsample_bytree=0.8, learning_rate=0.05, n_estimators=250),
    dict(max_depth=6, colsample_bytree=0.9, learning_rate=0.10, n_estimators=120, min_child_weight=20),
    dict(max_depth=7, colsample_bytree=0.4, learning_rate=0.10, n_estimators=120),
    dict(max_depth=7, colsample_bytree=0.7, learning_rate=0.08, n_estimators=150),
    dict(max_depth=7, colsample_bytree=0.8, learning_rate=0.05, n_estimators=250),
    dict(max_depth=8, colsample_bytree=0.6, learning_rate=0.06, n_estimators=200),
    dict(max_depth=9, colsample_bytree=0.5, learning_rate=0.05, n_estimators=200),
]
MEMBERS = [(cfg, s) for cfg in CONFIGS for s in SEEDS]
Xtr = _add_te(prepare(train), _keys, oof=True)
ytr = Y

t0 = time.time()
models = []
for i, (cfg, s) in enumerate(MEMBERS):
    m = xgb.XGBClassifier(**{**PARAMS, **cfg}, random_state=s)
    # each member sees a different random 70% of the training rows -> extra diversity
    sub = np.random.RandomState(1000 + i).choice(len(Xtr), size=int(0.7 * len(Xtr)), replace=False)
    m.fit(Xtr.iloc[sub], ytr[sub])
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  models={len(models)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = _add_te(prepare(df), _encode_keys(df), oof=False)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
