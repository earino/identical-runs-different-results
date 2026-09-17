"""Airline delay XGBoost with engineered features.

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


def _int_col(s: pd.Series) -> pd.Series:
    return s.str.slice(2).astype(int)


# category levels fitted on TRAIN only
_tr_prep_ref = None  # set below after prepare() is defined


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    month = _int_col(df["Month"])
    dom = _int_col(df["DayofMonth"])
    dow = _int_col(df["DayOfWeek"])
    dt = df["DepTime"]
    hour = dt // 100
    minute = dt % 100
    mins = hour * 60 + minute

    X["deptime"] = dt
    X["mins"] = mins
    X["dist"] = df["Distance"]
    X["logdist"] = np.log1p(df["Distance"])

    ang = 2.0 * np.pi
    X["h_sin"] = np.sin(ang * mins / 1440.0)
    X["h_cos"] = np.cos(ang * mins / 1440.0)
    X["m_sin"] = np.sin(ang * month / 12.0)
    X["m_cos"] = np.cos(ang * month / 12.0)
    X["d_sin"] = np.sin(ang * dom / 31.0)
    X["d_cos"] = np.cos(ang * dom / 31.0)
    X["w_sin"] = np.sin(ang * dow / 7.0)
    X["w_cos"] = np.cos(ang * dow / 7.0)

    X["late_night"] = ((dt >= 2400) | (hour <= 3)).astype(np.int8)
    X["early_morn"] = ((hour >= 4) & (hour <= 6)).astype(np.int8)

    X["month"] = month
    X["dom"] = dom
    X["dow"] = dow
    X["carrier"] = df["UniqueCarrier"]
    X["origin"] = df["Origin"]
    X["dest"] = df["Dest"]
    X["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["month_dow"] = month.astype(str) + "_" + dow.astype(str)
    X["hour"] = hour

    for c, levels in CATS.items():
        X[c] = pd.Categorical(X[c], categories=levels)
    return X


CAT_COLS = ["month", "dom", "dow", "carrier", "origin", "dest", "route", "month_dow", "hour"]
CATS = {}

# fit categorical levels on train only
_ref = train
_tmp = prepare(_ref)
CATS = {c: pd.Index(sorted(_tmp[c].dropna().unique().tolist())) for c in CAT_COLS}
del _tmp


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=400,
    learning_rate=0.08,
    max_depth=8,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare(train), to_y(train))
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
