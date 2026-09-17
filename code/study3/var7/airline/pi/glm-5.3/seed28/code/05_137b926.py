"""XGBoost binary classifier — airline dep delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).

Regime found on this time-shifted split (train 2005 -> eval 2006):
  - DEEP trees (d24) + heavy feature bagging (colsample_bytree 0.3) + slow lr (0.02, 200 rounds)
  - lean numerics beat cyclical encodings; redundant time encodings help under colsample
  - schedule-density counts (no labels) generalize; label TEs do NOT
  - 3-seed ensemble of the above.
All statistics fit on TRAIN only inside this module; prepare() applies them to any frame.
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
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEEDS = (42, 1, 7)


def _int_col(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _fit_stats(tr: pd.DataFrame) -> dict:
    hour = (tr["DepTime"] // 100).clip(0, 24)
    route = tr["Origin"].astype(str) + "_" + tr["Dest"].astype(str)
    h = hour.astype(str)
    return {
        "org": tr["Origin"].value_counts().to_dict(),
        "dest": tr["Dest"].value_counts().to_dict(),
        "route": route.value_counts().to_dict(),
        "car": tr["UniqueCarrier"].value_counts().to_dict(),
        "hour": hour.value_counts().to_dict(),
        "org_hour": (tr["Origin"].astype(str) + "_" + h).value_counts().to_dict(),
        "dest_hour": (tr["Dest"].astype(str) + "_" + h).value_counts().to_dict(),
        "route_hour": (route + "_" + h).value_counts().to_dict(),
        "car_hour": (tr["UniqueCarrier"].astype(str) + "_" + h).value_counts().to_dict(),
    }


def _make_prepare(train: pd.DataFrame):
    CNT = _fit_stats(train)
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
        X["month"] = mo
        X["dom"] = dom
        X["dow"] = dow
        X["hour"] = hour
        X["tmin"] = tmin
        X["minute"] = dep % 100
        X["tmin60"] = tmin / 60.0
        X["hour2"] = hour
        X["dep2"] = dep
        X["DepTime"] = dep
        X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
        X["logdist"] = np.log1p(X["Distance"])
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
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> missing
        return X

    return prepare


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
prepare = _make_prepare(train)

Xall = prepare(train)
yall = to_y(train)


def _make_model(seed: int, max_bin: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.02,
        max_depth=24,
        colsample_bytree=0.3,
        max_bin=max_bin,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


t0 = time.time()
models = []
for seed, max_bin in zip(SEEDS, (256, 384, 512)):
    m = _make_model(seed, max_bin)
    m.fit(Xall, yall)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
