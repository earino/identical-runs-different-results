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

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEEDS = [1, 2, 3, 4, 5]

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target encodings / counts (fit on training rows only) ---------------------
_y = to_y(train)
_prior = float(_y.mean())


def _te(keys_train: pd.Series, m: float) -> pd.Series:
    g = pd.DataFrame({"k": keys_train, "y": _y})
    stats = g.groupby("k")["y"].agg(["sum", "count"])
    return (stats["sum"] + m * _prior) / (stats["count"] + m)


def _cnum(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _hour_str(df: pd.DataFrame) -> pd.Series:
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    return ((dep // 100) % 24).astype(int).astype(str)


_tr_car = train["UniqueCarrier"].astype(str)
_tr_org = train["Origin"].astype(str)
_tr_dst = train["Dest"].astype(str)
te_carrier = _te(_tr_car, 10.0)
te_origin = _te(_tr_org, 10.0)
te_dest = _te(_tr_dst, 10.0)
te_car_h = _te(_tr_car + "_" + _hour_str(train), 30.0)
cnt_origin = _tr_org.value_counts()
cnt_dest = _tr_dst.value_counts()
cnt_route = (_tr_org + "_" + _tr_dst).value_counts()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering lives here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = (dep // 100) % 24
    mm = dep % 100
    tod = (hh * 60 + mm).astype(float)  # minutes since midnight
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["minute"] = mm.astype(float)
    month = _cnum(df["Month"]).astype(float)
    day = _cnum(df["DayofMonth"]).astype(float)
    dow = _cnum(df["DayOfWeek"]).astype(float)
    X["month"] = month
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["day"] = day
    X["dow"] = dow
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["distance"] = dist
    X["log_distance"] = np.log1p(dist)
    car = df["UniqueCarrier"].astype(str)
    org = df["Origin"].astype(str)
    dst = df["Dest"].astype(str)
    X["te_carrier"] = car.map(te_carrier).fillna(_prior).to_numpy()
    X["te_origin"] = org.map(te_origin).fillna(_prior).to_numpy()
    X["te_dest"] = dst.map(te_dest).fillna(_prior).to_numpy()
    X["te_car_h"] = (car + "_" + _hour_str(df)).map(te_car_h).fillna(_prior).to_numpy()
    X["cnt_origin"] = np.log1p(org.map(cnt_origin).fillna(0)).to_numpy()
    X["cnt_dest"] = np.log1p(dst.map(cnt_dest).fillna(0)).to_numpy()
    X["cnt_route"] = np.log1p((org + "_" + dst).map(cnt_route).fillna(0)).to_numpy()
    return X


# --- model: 5-seed XGBoost ensemble --------------------------------------------
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)

models = []
t0 = time.time()
for seed in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=8000,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        early_stopping_rounds=250,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    p = np.mean([m.predict_proba(prepare(df))[:, 1] for m in models], axis=0)
    return p


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
