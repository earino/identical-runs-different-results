"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: raw features + scheduled-departure time decomposition, fed to an XGBoost ensemble of
very deep (unlimited-depth) trees grown on bootstrap subsamples with column subsampling.  With
few boosting rounds this behaves like a boosted random forest: it captures high-order
interactions between carrier / origin / destination / time-of-day and generalises across the
2005 -> 2006 time split far better than shallow boosted trees.
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

# --- feature engineering state (fit on train only) ----------------------------
RAW_CAT = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in RAW_CAT}


def route_of(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _freq(series: pd.Series) -> dict:
    return (series.astype(str).value_counts() / len(series)).to_dict()


def _dep_hour(df: pd.DataFrame) -> np.ndarray:
    t = df["DepTime"].astype(float).to_numpy()
    return np.clip(np.floor(t / 100.0), 0, 23).astype(int)


def _hour_key(df: pd.DataFrame, name: str) -> pd.Series:
    h = pd.Series(_dep_hour(df), index=df.index).astype(str)
    if name == "origin_hour":
        return df["Origin"].astype(str) + "|" + h
    if name == "dest_hour":
        return df["Dest"].astype(str) + "|" + h
    if name == "route_hour":
        return route_of(df) + "|" + h
    return df["UniqueCarrier"].astype(str) + "|" + h


def _pair_key(df: pd.DataFrame, name: str) -> pd.Series:
    c = df["UniqueCarrier"].astype(str)
    if name == "carrier_origin":
        return c + "|" + df["Origin"].astype(str)
    return c + "|" + df["Dest"].astype(str)


freq_maps = {
    "route": _freq(route_of(train)),
    "origin": _freq(train["Origin"]),
    "dest": _freq(train["Dest"]),
    "carrier": _freq(train["UniqueCarrier"]),
    "origin_hour": _freq(_hour_key(train, "origin_hour")),
    "dest_hour": _freq(_hour_key(train, "dest_hour")),
    "route_hour": _freq(_hour_key(train, "route_hour")),
    "carrier_hour": _freq(_hour_key(train, "carrier_hour")),
    "carrier_origin": _freq(_pair_key(train, "carrier_origin")),
    "carrier_dest": _freq(_pair_key(train, "carrier_dest")),
}


def _raw_key(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "route":
        return route_of(df)
    if name == "origin":
        return df["Origin"].astype(str)
    if name == "dest":
        return df["Dest"].astype(str)
    if name == "carrier":
        return df["UniqueCarrier"].astype(str)
    if name in ("carrier_origin", "carrier_dest"):
        return _pair_key(df, name)
    return _hour_key(df, name)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[["DepTime", "Distance"]].copy()
    for c in RAW_CAT:
        X[c] = pd.Categorical(df[c].values, categories=cat_levels[c])

    t = df["DepTime"].astype(float).to_numpy()
    hh = np.floor(t / 100.0)
    mm = t % 100.0
    mins = hh * 60.0 + mm
    X["dep_mins"] = mins
    X["dep_sin"] = np.sin(2.0 * np.pi * (mins % 1440.0) / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * (mins % 1440.0) / 1440.0)
    X["dep_minute"] = mm
    X["is_round"] = (mm == 0).astype(np.int8)
    X["is_round5"] = (mm % 5 == 0).astype(np.int8)
    X["is_round15"] = (mm % 15 == 0).astype(np.int8)

    dow = pd.to_numeric(df["DayOfWeek"].astype(str).str.split("-").str[-1], errors="coerce").to_numpy()
    X["is_weekend"] = np.isin(dow, [6, 7]).astype(np.int8)

    for name in freq_maps:
        X[name + "_freq"] = _raw_key(df, name).map(freq_maps[name]).fillna(0.0).to_numpy()
    X["log_distance"] = np.log1p(X["Distance"].to_numpy())
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: bagged ensemble of deep trees -------------------------------------
N_MODELS = 16
PARAMS = dict(
    n_estimators=30,
    max_depth=0,               # unlimited depth
    learning_rate=0.08,
    subsample=0.95,
    colsample_bytree=0.42,
    min_child_weight=1,
    max_cat_to_onehot=64,
    max_bin=512,
    tree_method="hist",
    enable_categorical=True,
    n_jobs=N_JOBS,
)

X_train = prepare(train)
y_train = to_y(train)

models = []
t0 = time.time()
for k in range(N_MODELS):
    m = xgb.XGBClassifier(random_state=SEED + k, **PARAMS)
    m.fit(X_train, y_train)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    proba = np.zeros(len(X), dtype=np.float64)
    for m in models:
        proba += m.predict_proba(X)[:, 1]
    return proba / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
