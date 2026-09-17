"""XGBoost binary classifier with engineered features.

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

# --- feature engineering ------------------------------------------------------
_o = train["Origin"].astype(str)
_d = train["Dest"].astype(str)
_c = train["UniqueCarrier"].astype(str)
_dep = pd.to_numeric(train["DepTime"], errors="coerce").fillna(-1).astype(int)
_h = (_dep // 100).clip(0, 23)
def _arr_hour(dep_series, dist_series):
    dep = pd.to_numeric(dep_series, errors="coerce").fillna(-1).astype(int)
    h = (dep // 100).clip(0, 23)
    m = (dep % 100).clip(0, 59)
    dist = pd.to_numeric(dist_series, errors="coerce").fillna(0)
    return ((h * 60 + m + dist * 0.12 + 30) // 60).astype(int) % 24

ORIG_TOTAL = _o.value_counts()
ORIG_HOUR = pd.DataFrame({"o": _o, "h": _h}).groupby(["o", "h"]).size()
CARR_TOTAL = _c.value_counts()
CARR_HOUR = pd.DataFrame({"c": _c, "h": _h}).groupby(["c", "h"]).size()
DEST_TOTAL = _d.value_counts()
DEST_HOUR = pd.DataFrame({"d": _d, "h": _h}).groupby(["d", "h"]).size()
_ah = _arr_hour(train["DepTime"], train["Distance"])
DEST_ARR_HOUR = pd.DataFrame({"d": _d, "a": _ah}).groupby(["d", "a"]).size()
ORIG_ARR_HOUR = pd.DataFrame({"o": _o, "a": _ah}).groupby(["o", "a"]).size()
CARR_ORIG = pd.DataFrame({"c": _c, "o": _o}).groupby(["c", "o"]).size()
CARR_DEST = pd.DataFrame({"c": _c, "d": _d}).groupby(["c", "d"]).size()


def _cat_int(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("c-", "", regex=False).astype(int)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    month = _cat_int(df["Month"])
    dow = _cat_int(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(-1).astype(int)
    hour = (dep // 100).clip(0, 23)
    minute = (dep % 100).clip(0, 59)
    X["month"] = month
    X["day"] = _cat_int(df["DayofMonth"])
    X["dow"] = dow
    X["hour"] = hour
    X["minute"] = minute
    X["dep_raw"] = dep
    X["is_weekend"] = (dow >= 6).astype(int)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["distance"].clip(lower=0))
    X["origin"] = df["Origin"].astype(str)
    X["dest"] = df["Dest"].astype(str)
    X["carrier"] = df["UniqueCarrier"].astype(str)
    oh = pd.MultiIndex.from_arrays([X["origin"].values, hour.values])
    X["orig_hour_share"] = ORIG_HOUR.reindex(oh).fillna(0).to_numpy() / X["origin"].map(ORIG_TOTAL).fillna(1).to_numpy()
    ch = pd.MultiIndex.from_arrays([X["carrier"].values, hour.values])
    X["carr_hour_share"] = CARR_HOUR.reindex(ch).fillna(0).to_numpy() / X["carrier"].map(CARR_TOTAL).fillna(1).to_numpy()
    dh = pd.MultiIndex.from_arrays([X["dest"].values, hour.values])
    X["dest_hour_share"] = DEST_HOUR.reindex(dh).fillna(0).to_numpy() / X["dest"].map(DEST_TOTAL).fillna(1).to_numpy()
    X["orig_hour_count"] = np.log1p(ORIG_HOUR.reindex(oh).fillna(0).to_numpy())
    X["dest_hour_count"] = np.log1p(DEST_HOUR.reindex(dh).fillna(0).to_numpy())
    co = pd.MultiIndex.from_arrays([X["carrier"].values, X["origin"].values])
    X["carr_orig_share"] = CARR_ORIG.reindex(co).fillna(0).to_numpy() / X["carrier"].map(CARR_TOTAL).fillna(1).to_numpy()
    cd2 = pd.MultiIndex.from_arrays([X["carrier"].values, X["dest"].values])
    X["carr_dest_share"] = CARR_DEST.reindex(cd2).fillna(0).to_numpy() / X["carrier"].map(CARR_TOTAL).fillna(1).to_numpy()
    ah = _arr_hour(df["DepTime"], df["Distance"])
    X["arr_hour"] = ah
    da = pd.MultiIndex.from_arrays([X["dest"].values, ah.values])
    X["dest_arr_hour_share"] = DEST_ARR_HOUR.reindex(da).fillna(0).to_numpy() / X["dest"].map(DEST_TOTAL).fillna(1).to_numpy()
    oa = pd.MultiIndex.from_arrays([X["origin"].values, ah.values])
    X["orig_arr_hour_share"] = ORIG_ARR_HOUR.reindex(oa).fillna(0).to_numpy() / X["origin"].map(ORIG_TOTAL).fillna(1).to_numpy()
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return X


DERIVED = derive(train)
NUM_COLS = [
    "day", "hour", "minute", "dep_raw", "is_weekend", "distance", "log_distance",
    "hour_sin", "hour_cos", "orig_hour_share", "carr_hour_share",
    "dest_hour_share", "orig_hour_count", "dest_hour_count",
    "carr_orig_share", "carr_dest_share", "arr_hour",
    "dest_arr_hour_share", "orig_arr_hour_share",
]
CAT_COLS = ["month", "dow", "origin", "dest", "carrier"]
cat_levels = {c: pd.Index(sorted(DERIVED[c].dropna().unique())) for c in CAT_COLS}


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Y_TRAIN = to_y(train)
PRIOR = float(Y_TRAIN.mean())

# --- target encoding of time-conditioned groups (OOF for training) ------------
TE_NAMES = ["oh", "dh", "ch"]
TE_SMOOTH = 50.0


def _te_keys(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "oh": X["origin"].astype(str) + "|" + X["hour"].astype(str),
            "dh": X["dest"].astype(str) + "|" + X["hour"].astype(str),
            "ch": X["carrier"].astype(str) + "|" + X["hour"].astype(str),
        },
        index=X.index,
    )


def _te_map(y: np.ndarray, keys: pd.Series, k: float, prior: float) -> pd.Series:
    g = pd.DataFrame({"k": keys.values, "y": y}).groupby("k")["y"].agg(["mean", "count"])
    return (g["mean"] * g["count"] + k * prior) / (g["count"] + k)


_KEYS_TRAIN = _te_keys(DERIVED)
TE_MAP = {n: _te_map(Y_TRAIN, _KEYS_TRAIN[n], TE_SMOOTH, PRIOR) for n in TE_NAMES}
TE_FEATS = [n + "_te" for n in TE_NAMES]
ALL_COLS = NUM_COLS + CAT_COLS + TE_FEATS


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = derive(df)
    keys = _te_keys(X)
    for n in TE_NAMES:
        X[n + "_te"] = keys[n].map(TE_MAP[n]).fillna(PRIOR).astype(float)
    X = X[ALL_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def train_features() -> pd.DataFrame:
    keys = _KEYS_TRAIN
    oof = {n: np.full(len(DERIVED), PRIOR) for n in TE_NAMES}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, va in kf.split(DERIVED):
        prior = float(Y_TRAIN[tr].mean())
        for n in TE_NAMES:
            m = _te_map(Y_TRAIN[tr], keys[n].iloc[tr], TE_SMOOTH, prior)
            oof[n][va] = keys[n].iloc[va].map(m).fillna(prior).to_numpy()
    X = DERIVED[NUM_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    for n in TE_NAMES:
        X[n + "_te"] = oof[n]
    return X[ALL_COLS]


# --- model ensemble -----------------------------------------------------------
CONFIGS = [
    dict(n_estimators=150, max_depth=6, learning_rate=0.1),
    dict(n_estimators=150, max_depth=6, learning_rate=0.1),
    dict(n_estimators=120, max_depth=7, learning_rate=0.1),
    dict(n_estimators=120, max_depth=7, learning_rate=0.1),
    dict(n_estimators=250, max_depth=6, learning_rate=0.06),
    dict(n_estimators=250, max_depth=6, learning_rate=0.06),
    dict(n_estimators=100, max_depth=8, learning_rate=0.1),
    dict(n_estimators=100, max_depth=8, learning_rate=0.1),
    dict(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=150, max_depth=7, learning_rate=0.08, subsample=0.8, min_child_weight=3),
    dict(n_estimators=400, max_depth=6, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7, min_child_weight=8),
    dict(n_estimators=80, max_depth=10, learning_rate=0.1),
    dict(n_estimators=150, max_depth=6, learning_rate=0.1),
    dict(n_estimators=300, max_depth=0, max_leaves=64, learning_rate=0.08, grow_policy="lossguide", subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=200, max_depth=0, max_leaves=32, learning_rate=0.1, grow_policy="lossguide"),
]

X_train = train_features()
y_train = to_y(train)
models = []
t0 = time.time()
for i, cfg in enumerate(CONFIGS):
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED + 7 * i,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
