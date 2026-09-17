"""XGBoost binary classifier — airline dep delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
All feature engineering lives inside prepare(); all statistics fit on TRAIN only.
Key regime found: DEEP trees (d24) + heavy feature bagging (colsample 0.3) + slow lr
beats shallow/full-feature boosting on this time-shifted split.
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


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _int_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


# --- train-fitted statistics (schedule density; NO labels involved) --------------
_hour_tr = (train["DepTime"] // 100).clip(0, 24)
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_h = _hour_tr.astype(str)
CNT = {
    "org": train["Origin"].value_counts().to_dict(),
    "dest": train["Dest"].value_counts().to_dict(),
    "route": _route_tr.value_counts().to_dict(),
    "car": train["UniqueCarrier"].value_counts().to_dict(),
    "hour": _hour_tr.value_counts().to_dict(),
    "org_hour": (train["Origin"].astype(str) + "_" + _h).value_counts().to_dict(),
    "dest_hour": (train["Dest"].astype(str) + "_" + _h).value_counts().to_dict(),
    "route_hour": (_route_tr + "_" + _h).value_counts().to_dict(),
    "car_hour": (train["UniqueCarrier"].astype(str) + "_" + _h).value_counts().to_dict(),
}
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    mo = _int_col(df["Month"])
    dom = _int_col(df["DayofMonth"])
    dow = _int_col(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce")
    hour = (dep // 100).clip(0, 24)
    tmin = hour * 60 + dep % 100
    doy = (mo - 1) * 31 + dom
    X["month"] = mo
    X["dom"] = dom
    X["dow"] = dow
    X["hour"] = hour
    X["tmin"] = tmin
    X["tmin_sin"] = np.sin(2 * np.pi * tmin / 1440.0)
    X["tmin_cos"] = np.cos(2 * np.pi * tmin / 1440.0)
    X["DepTime"] = dep
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["logdist"] = np.log1p(X["Distance"])
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    hs = hour.astype(str)
    X["cnt_org"] = np.log1p(df["Origin"].map(CNT["org"]).fillna(0))
    X["cnt_dest"] = np.log1p(df["Dest"].map(CNT["dest"]).fillna(0))
    X["cnt_route"] = np.log1p(route.map(CNT["route"]).fillna(0))
    X["cnt_car"] = np.log1p(df["UniqueCarrier"].map(CNT["car"]).fillna(0))
    X["cnt_hour"] = np.log1p(hs.map(CNT["hour"]).fillna(0))
    X["cnt_org_hour"] = np.log1p((df["Origin"].astype(str) + "_" + hs).map(CNT["org_hour"]).fillna(0))
    X["cnt_dest_hour"] = np.log1p((df["Dest"].astype(str) + "_" + hs).map(CNT["dest_hour"]).fillna(0))
    X["cnt_route_hour"] = np.log1p((route + "_" + hs).map(CNT["route_hour"]).fillna(0))
    X["cnt_car_hour"] = np.log1p((df["UniqueCarrier"].astype(str) + "_" + hs).map(CNT["car_hour"]).fillna(0))
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen -> missing
    return X


# --- model ---------------------------------------------------------------------
Xall = prepare(train)
yall = to_y(train)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.02,
    max_depth=24,
    colsample_bytree=0.3,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xall, yall)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
