"""XGBoost binary classifier for flight delay prediction.

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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

BASE_CATS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
_dist_train = pd.to_numeric(train["Distance"], errors="coerce")
DIST_EDGES = list(pd.qcut(_dist_train, 10, retbins=True, duplicates="drop")[1])
DIST_EDGES[0] = -1
DIST_EDGES[-1] = DIST_EDGES[-1] + 1

# --- helpers ------------------------------------------------------------------
def _parse_code(s: pd.Series) -> np.ndarray:
    v = s.astype(str).str.replace("c-", "", regex=False)
    return pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)


def _hour(df: pd.DataFrame) -> np.ndarray:
    dep = pd.to_numeric(df["DepTime"], errors="coerce").to_numpy(dtype=float)
    h = dep // 100
    bad = ~((dep >= 0) & (dep <= 2359)) | (h < 0) | (h > 23)
    h = np.where(bad, np.nan, h)
    return h


def _dist_bin(df: pd.DataFrame) -> np.ndarray:
    d = pd.to_numeric(df["Distance"], errors="coerce")
    b = pd.cut(d, bins=DIST_EDGES, labels=False).fillna(-1).astype(int)
    return b.to_numpy()


def _ch(df: pd.DataFrame) -> pd.Series:
    return df["UniqueCarrier"].astype(str) + "_" + np.where(np.isnan(_hour(df)), -1, _hour(df)).astype(int).astype(str)


def _hdb(df: pd.DataFrame) -> pd.Series:
    h = pd.Series(np.where(np.isnan(_hour(df)), -1, _hour(df)).astype(int).astype(str), index=df.index)
    d = pd.Series(_dist_bin(df).astype(str), index=df.index)
    return h + "_" + d


# --- category levels fit on training data only --------------------------------
BASE_LEVELS = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in BASE_CATS}
CH_LEVELS = pd.Index(sorted(_ch(train).unique()))
HDB_LEVELS = pd.Index(sorted(_hdb(train).unique()))
_origin_counts = train["Origin"].astype(str).value_counts()
_dest_counts = train["Dest"].astype(str).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in BASE_CATS:
        X[c] = pd.Categorical(df[c].astype(str), categories=BASE_LEVELS[c])

    dep = pd.to_numeric(df["DepTime"], errors="coerce").to_numpy(dtype=float)
    hour = _hour(df)
    minute = dep % 100
    minute = np.where((dep >= 0) & (dep <= 2359) & (minute >= 0) & (minute <= 59), minute, np.nan)
    tod = hour * 60 + minute
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)

    dow = _parse_code(df["DayOfWeek"])
    X["weekend"] = (dow >= 6).astype(float)
    X["dow_num"] = dow
    X["day_num"] = _parse_code(df["DayofMonth"])
    X["month_num"] = _parse_code(df["Month"])
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["logd"] = np.log1p(X["Distance"])

    X["ch"] = pd.Categorical(_ch(df), categories=CH_LEVELS)
    X["hdb"] = pd.Categorical(_hdb(df), categories=HDB_LEVELS)
    X["origin_count"] = df["Origin"].astype(str).map(_origin_counts).fillna(0).to_numpy()
    X["dest_count"] = df["Dest"].astype(str).map(_dest_counts).fillna(0).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: ensemble of XGBoost models at several depths ----------------------
def make_model(depth: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=700,
        max_depth=depth,
        learning_rate=0.05,
        max_cat_to_onehot=32,
        tree_method="hist",
        enable_categorical=True,
        random_state=SEED,
        n_jobs=N_JOBS,
    )


MODEL_DEPTHS = [1, 6, 11, 16]
models = []
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
for depth in MODEL_DEPTHS:
    m = make_model(depth)
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
