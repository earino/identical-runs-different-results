"""XGBoost binary classifier for airline dep_delayed_15min. The only file the agent edits.

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

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
TE_COLS = ["UniqueCarrier", "Origin", "Dest"]  # target-encoded (smoothed)
TE_M = 100  # TE smoothing strength
RAW_COLS = ["DepTime", "Distance"]
feature_cols = CAT_COLS + RAW_COLS
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

_ytr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
_prior = _ytr.mean()
_g = pd.DataFrame({"k": train["UniqueCarrier"].values, "y": _ytr}).groupby("k")["y"].agg(["sum", "count"])
# smoothed target-encoding maps, fit on TRAIN ONLY (module level), applied inside prepare()
# frequency (busyness) maps, fit on TRAIN ONLY
FREQ = {}
FREQ["Origin"] = train["Origin"].value_counts().to_dict()
FREQ["Dest"] = train["Dest"].value_counts().to_dict()
FREQ["oh"] = (train["Origin"] + "_" + (np.minimum(train["DepTime"].astype(int) // 100, 24)).astype(str)).value_counts().to_dict()
_trd = train.copy()
_trd["route"] = _trd["Origin"] + "_" + _trd["Dest"]
_trd["ch"] = _trd["UniqueCarrier"] + "_" + (np.minimum(_trd["DepTime"].astype(int) // 100, 24)).astype(str)
_trd["dh"] = _trd["Dest"] + "_" + (np.minimum(_trd["DepTime"].astype(int) // 100, 24)).astype(str)
_trd["odow"] = _trd["Origin"] + "_" + _trd["DayOfWeek"]
for _k, _col in [("route", "route"), ("carrier", "UniqueCarrier"), ("ch", "ch"), ("dh", "dh"), ("odow", "odow")]:
    FREQ[_k] = _trd[_col].value_counts().to_dict()
TE_MAPS = {}
for c in TE_COLS:
    g = pd.DataFrame({"k": train[c].values, "y": _ytr}).groupby("k")["y"].agg(["sum", "count"])
    TE_MAPS[c] = (((g["sum"] + TE_M * _prior) / (g["count"] + TE_M)).to_dict(), _prior)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[RAW_COLS + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dt = X["DepTime"].fillna(0).astype(int)
    hour = np.minimum(dt // 100, 24)
    tod = hour * 60 + dt % 100
    X["hour"] = hour
    X["minute"] = dt % 100
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    for c in TE_COLS:
        mp, prior = TE_MAPS[c]
        X[c + "_te"] = df[c].map(mp).fillna(prior)
    hour_str = hour.astype(str)
    X["origin_cnt"] = df["Origin"].map(FREQ["Origin"]).fillna(0).astype(float)
    X["dest_cnt"] = df["Dest"].map(FREQ["Dest"]).fillna(0).astype(float)
    X["oh_cnt"] = (df["Origin"] + "_" + hour_str).map(FREQ["oh"]).fillna(0).astype(float)
    d2 = df.assign(_h=hour_str)
    X["route_cnt"] = (d2["Origin"] + "_" + d2["Dest"]).map(FREQ["route"]).fillna(0).astype(float)
    X["carrier_cnt"] = df["UniqueCarrier"].map(FREQ["carrier"]).fillna(0).astype(float)
    X["ch_cnt"] = (df["UniqueCarrier"] + "_" + hour_str).map(FREQ["ch"]).fillna(0).astype(float)
    X["dh_cnt"] = (df["Dest"] + "_" + hour_str).map(FREQ["dh"]).fillna(0).astype(float)
    X["odow_cnt"] = (df["Origin"] + "_" + df["DayOfWeek"]).map(FREQ["odow"]).fillna(0).astype(float)
    # traffic shares: hour-level volume relative to base volume (year-stable schedule patterns)
    X["oh_share"] = X["oh_cnt"] / X["origin_cnt"].replace(0, np.nan)
    X["ch_share"] = X["ch_cnt"] / X["carrier_cnt"].replace(0, np.nan)
    X["dh_share"] = X["dh_cnt"] / X["dest_cnt"].replace(0, np.nan)
    X["route_share"] = X["route_cnt"] / (X["origin_cnt"] + X["dest_cnt"]).replace(0, np.nan)
    X["odow_share"] = X["odow_cnt"] / X["origin_cnt"].replace(0, np.nan)
    num = X.select_dtypes(include=[np.number]).columns
    X[num] = X[num].fillna(-1.0)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
# depth-diverse bagged pair of XGBoost models (probabilities averaged)
X = prepare(train)
y = to_y(train)
ENSEMBLE = [
    dict(n_estimators=1200, learning_rate=0.035, max_depth=20, colsample_bytree=0.25, reg_alpha=4, random_state=7),
    dict(n_estimators=1200, learning_rate=0.035, max_depth=24, colsample_bytree=0.25, reg_alpha=4, random_state=13),
]
models = []
t0 = time.time()
for cfg in ENSEMBLE:
    m = xgb.XGBClassifier(
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **cfg,
    )
    m.fit(X, y)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    p = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
