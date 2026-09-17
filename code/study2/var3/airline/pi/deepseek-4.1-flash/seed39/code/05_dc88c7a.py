"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
"""
import json
import os

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

CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def _hhmm(df: pd.DataFrame) -> pd.Series:
    """Scheduled departure hhmm as minutes-since-midnight (24xx -> 00xx)."""
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    return t.where(t < 2400, t - 2400)


def _hour(hhmm: pd.Series) -> pd.Series:
    return (hhmm // 100).clip(0, 23)


# ---- train-fit lookups (target-free) ----------------------------------------
_h_tr = _hour(_hhmm(train))
_o_key_tr = train["Origin"] + "_" + _h_tr.astype(str)
_d_key_tr = train["Dest"] + "_" + _h_tr.astype(str)
_o_h_cnt = _o_key_tr.value_counts()
_d_h_cnt = _d_key_tr.value_counts()
_o_h_ndest = train.groupby(_o_key_tr)["Dest"].nunique()
_d_h_norig = train.groupby(_d_key_tr)["Origin"].nunique()
_o_h_dist = train.assign(_k=_o_key_tr).groupby("_k")["Distance"].mean()
_d_h_dist = train.assign(_k=_d_key_tr).groupby("_k")["Distance"].mean()
_o_tot = train["Origin"].value_counts()
_d_tot = train["Dest"].value_counts()
_carrier_freq = train["UniqueCarrier"].value_counts(normalize=True)
_origin_freq = _o_tot / len(train)
_dest_freq = _d_tot / len(train)
_route_freq = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str)).value_counts(normalize=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = df[CAT_COLS + ["Distance"]].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=CAT_LEVELS[c])  # unseen levels -> NaN
    hhmm = _hhmm(df)
    hour = _hour(hhmm)
    minute = (hhmm % 100).clip(0, 59)
    tod = hour + minute / 60.0
    X["hour"] = hour
    X["minute"] = minute
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)
    X["q15"] = (tod * 4).round()
    X["DepTime"] = pd.to_numeric(df["DepTime"], errors="coerce")
    X["logd"] = np.log1p(X["Distance"])

    # train-fit frequency encodings
    X["f_carrier"] = df["UniqueCarrier"].map(_carrier_freq).astype(float)
    X["f_origin"] = df["Origin"].map(_origin_freq).astype(float)
    X["f_dest"] = df["Dest"].map(_dest_freq).astype(float)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["f_route"] = route.map(_route_freq).astype(float)

    # airport-hour congestion (target-free): how busy is this airport at this hour
    ok = df["Origin"] + "_" + hour.astype(str)
    dk = df["Dest"] + "_" + hour.astype(str)
    o_cnt, d_cnt = ok.map(_o_h_cnt), dk.map(_d_h_cnt)
    X["o_h"] = o_cnt.fillna(0)
    X["d_h"] = d_cnt.fillna(0)
    X["o_share"] = (o_cnt / df["Origin"].map(_o_tot)).astype(float)
    X["d_share"] = (d_cnt / df["Dest"].map(_d_tot)).astype(float)
    X["o_h_nd"] = ok.map(_o_h_ndest).fillna(0)
    X["d_h_no"] = dk.map(_d_h_norig).fillna(0)
    X["o_h_dist"] = ok.map(_o_h_dist).astype(float)
    X["d_h_dist"] = dk.map(_d_h_dist).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=24,
    learning_rate=0.011,
    colsample_bytree=0.2,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

model.fit(prepare(train), to_y(train))


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


if __name__ == "__main__":
    evald = pd.read_csv("data/eval.csv")
    eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
    print(f"Eval AUC: {eval_auc:.4f}")
