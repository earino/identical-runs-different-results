"""XGBoost binary classifier for airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract:
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. Module-level `predict_proba(df)`: raw DataFrame -> 1-D array of P(positive).
     All feature engineering lives inside prepare(); statistics fit on train.csv only.

Model: 3-member XGBoost ensemble. Each member trains on train.csv augmented with a copy
whose DepTime is jittered by +-10 minutes (label-preserving smoothing of time-of-day
buckets; combats 2005->2006 shift). Members differ in depth/lambda/one-hot/seeds.
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
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- statistics fit on the training data only ----------------------------------
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ["UniqueCarrier", "Origin", "Dest"]}
_tr_hour = (train["DepTime"].astype(int) // 100) % 24
HOUR_CARR_CATS = pd.Index(sorted((_tr_hour.astype(str) + "_" + train["UniqueCarrier"].astype(str)).unique()))


_tr_h = (train["DepTime"].astype(int) // 100) % 24
_tr_m = _tr_h * 60 + (train["DepTime"].astype(int) % 100)
OH_COUNTS = pd.DataFrame({"k": train["Origin"].astype(str) + "_" + _tr_h.astype(str)}).groupby("k").size()
DO_COUNTS = pd.DataFrame({"k": train["Dest"].astype(str) + "_" + _tr_h.astype(str)}).groupby("k").size()
O96_COUNTS = pd.DataFrame({"k": train["Origin"].astype(str) + "_" + (_tr_m // 15).astype(str)}).groupby("k").size()
D96_COUNTS = pd.DataFrame({"k": train["Dest"].astype(str) + "_" + (_tr_m // 15).astype(str)}).groupby("k").size()
CH_COUNTS = pd.DataFrame({"k": train["UniqueCarrier"].astype(str) + "_" + _tr_h.astype(str)}).groupby("k").size()
OC_COUNTS = pd.DataFrame({"k": train["Origin"].astype(str) + "_" + train["UniqueCarrier"].astype(str)}).groupby("k").size()
DC_COUNTS = pd.DataFrame({"k": train["Dest"].astype(str) + "_" + train["UniqueCarrier"].astype(str)}).groupby("k").size()
_route = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
RS = pd.DataFrame({"k": _route, "s": np.sin(2 * np.pi * _tr_m / 1440), "c": np.cos(2 * np.pi * _tr_m / 1440)}).groupby("k").agg("mean")
RMED = pd.DataFrame({"k": _route, "m": _tr_m}).groupby("k")["m"].median()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dep = df["DepTime"].astype(int)
    hour = (dep // 100) % 24
    mins = hour * 60 + dep % 100
    X["DepTime"] = dep
    X["Distance"] = df["Distance"].astype(float)
    X["logDist"] = np.log1p(X["Distance"])
    X["Month"] = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DayofMonth"] = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DayOfWeek"] = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(int)
    X["DepHour"] = hour
    X["DepMinute"] = dep % 100
    X["MinsOfDay"] = mins
    X["sin_tod"] = np.sin(2 * np.pi * mins / 1440.0)
    X["cos_tod"] = np.cos(2 * np.pi * mins / 1440.0)
    X["UniqueCarrier"] = pd.Categorical(df["UniqueCarrier"], categories=cat_levels["UniqueCarrier"])
    X["Origin"] = pd.Categorical(df["Origin"], categories=cat_levels["Origin"])
    X["Dest"] = pd.Categorical(df["Dest"], categories=cat_levels["Dest"])
    X["HourCat"] = pd.Categorical(hour, categories=pd.Index(range(24)))
    X["TodCat"] = pd.Categorical((mins // 15).astype(int), categories=pd.Index(range(96)))
    X["HourCarr"] = pd.Categorical(
        hour.astype(str) + "_" + df["UniqueCarrier"].astype(str), categories=HOUR_CARR_CATS
    )
    X["cong_o"] = np.log1p(
        (df["Origin"].astype(str) + "_" + hour.astype(str)).map(OH_COUNTS).fillna(0)
    ).to_numpy()
    X["cong_d"] = np.log1p(
        (df["Dest"].astype(str) + "_" + hour.astype(str)).map(DO_COUNTS).fillna(0)
    ).to_numpy()
    _t = (mins // 15).astype(str)
    X["cong_o96"] = np.log1p(
        (df["Origin"].astype(str) + "_" + _t).map(O96_COUNTS).fillna(0)
    ).to_numpy()
    X["cong_d96"] = np.log1p(
        (df["Dest"].astype(str) + "_" + _t).map(D96_COUNTS).fillna(0)
    ).to_numpy()
    X["cong_ch"] = np.log1p(
        (df["UniqueCarrier"].astype(str) + "_" + hour.astype(str)).map(CH_COUNTS).fillna(0)
    ).to_numpy()
    X["hub_o"] = np.log1p(
        (df["Origin"].astype(str) + "_" + df["UniqueCarrier"].astype(str)).map(OC_COUNTS).fillna(0)
    ).to_numpy()
    X["hub_d"] = np.log1p(
        (df["Dest"].astype(str) + "_" + df["UniqueCarrier"].astype(str)).map(DC_COUNTS).fillna(0)
    ).to_numpy()
    _r = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["route_sin"] = _r.map(RS["s"]).fillna(0).to_numpy()
    X["route_cos"] = _r.map(RS["c"]).fillna(0).to_numpy()
    _med = _r.map(RMED).fillna(720).to_numpy()
    X["route_dep_dev"] = (mins.to_numpy() - _med + 720) % 1440 - 720
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def jitter_df(seed: int, amp: int) -> pd.DataFrame:
    mins = ((train["DepTime"].astype(int) // 100) % 24) * 60 + train["DepTime"].astype(int) % 100
    rng = np.random.RandomState(seed)
    j = (mins + rng.randint(-amp, amp + 1, len(mins))) % 1440
    out = train.copy()
    out["DepTime"] = (j // 60) * 100 + j % 60
    return out


BASE = dict(learning_rate=0.05, min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
            reg_lambda=100, gamma=0.5)
MEMBERS = [
    (dict(n_estimators=900, max_depth=8, **BASE), 42, 7),    # A: base config
    (dict(n_estimators=600, max_depth=10, reg_lambda=30, learning_rate=0.03, min_child_weight=20,
          subsample=0.7, colsample_bytree=0.7, gamma=0.5), 4, 8),  # C: deeper, lighter L2
    (dict(n_estimators=900, max_depth=8, max_cat_to_onehot=96, **BASE), 5, 12),  # D: one-hot time buckets
]

ytr = to_y(train)
models = []
t0 = time.time()
for params, seed, jseed in MEMBERS:
    X = pd.concat([prepare(train), prepare(jitter_df(jseed, 10))], ignore_index=True)
    m = xgb.XGBClassifier(
        tree_method="hist", enable_categorical=True, random_state=seed, n_jobs=N_JOBS, **params
    )
    m.fit(X, np.tile(ytr, 2))
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s for {len(models)} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
