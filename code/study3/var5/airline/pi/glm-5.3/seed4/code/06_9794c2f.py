"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering (all statistics fitted on TRAIN only) ------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}

_hour_tr = (train["DepTime"].astype(int) // 100) % 24
_tod_tr = (_hour_tr * 60 + train["DepTime"].astype(int) % 100).to_numpy()
_route_tr = train["Origin"].astype(str) + "_" + train["Dest"].astype(str)
_train_counts = {
    "oh": (train["Origin"].astype(str) + "@" + _hour_tr.astype(str)).value_counts(),
    "dh": (train["Dest"].astype(str) + "@" + _hour_tr.astype(str)).value_counts(),
    "route": _route_tr.value_counts(),
    "origin": train["Origin"].astype(str).value_counts(),
    "dest": train["Dest"].astype(str).value_counts(),
    "co": (train["UniqueCarrier"].astype(str) + "@" + train["Origin"].astype(str)).value_counts(),
    "cd": (train["UniqueCarrier"].astype(str) + "@" + train["Dest"].astype(str)).value_counts(),
}
_route_carriers = train.groupby(_route_tr)["UniqueCarrier"].nunique()

# schedule-position CDFs fitted on train:
#  origin: rank of this departure time within the origin's daily departure distribution
#  dest:   rank of the estimated arrival time within the dest's arrival distribution
_ARR_TRAVEL_MIN = 40.0  # taxi + climb overhead
_ARR_SPEED_MPH = 450.0
_arr_tr = (_tod_tr + _ARR_TRAVEL_MIN + train["Distance"].astype(float).to_numpy() / _ARR_SPEED_MPH * 60) % 1440
org_tod = {o: np.sort(g.to_numpy()) for o, g in pd.Series(_tod_tr).groupby(train["Origin"].astype(str).values)}
dst_arr = {d: np.sort(g.to_numpy()) for d, g in pd.Series(_arr_tr).groupby(train["Dest"].astype(str).values)}


def _cnt(key: "pd.Series[str]", name: str) -> np.ndarray:
    return np.log1p(key.map(_train_counts[name]).fillna(0).to_numpy(dtype=float))


def _cdf(sorted_ref: dict, keys: "pd.Series[str]", tods: np.ndarray) -> np.ndarray:
    """Per-key empirical CDF of `tods` inside the train-fitted sorted arrays (vectorized per key)."""
    out = np.full(len(tods), 0.5)
    ks = pd.Series(keys.to_numpy(), index=np.arange(len(keys)))
    for k, idx in ks.groupby(ks).groups.items():
        v = sorted_ref.get(k)
        if v is None:
            continue
        pos = np.asarray(idx)
        out[pos] = np.searchsorted(v, tods[pos]) / len(v)
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # calendar fields come as c-<n> strings -> numeric
    X["month"] = df["Month"].astype(str).str[2:].astype(int)
    X["day"] = df["DayofMonth"].astype(str).str[2:].astype(int)
    X["dow"] = df["DayOfWeek"].astype(str).str[2:].astype(int)
    # scheduled departure time hhmm -> hour / minute / time-of-day (wraps hour 24/26 -> 0/2)
    dep = df["DepTime"].astype(int)
    X["DepTime"] = dep
    hour = (dep // 100) % 24
    X["hour"] = hour
    X["minute"] = dep % 100
    X["tod"] = hour * 60 + X["minute"]
    # geography / carrier
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    X["Distance"] = df["Distance"].astype(float)
    # congestion counts (train-fitted)
    X["cnt_oh"] = _cnt(df["Origin"].astype(str) + "@" + hour.astype(str), "oh")
    X["cnt_dh"] = _cnt(df["Dest"].astype(str) + "@" + hour.astype(str), "dh")
    route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    X["cnt_route"] = _cnt(route, "route")
    X["cnt_origin"] = _cnt(df["Origin"].astype(str), "origin")
    X["cnt_dest"] = _cnt(df["Dest"].astype(str), "dest")
    X["cnt_co"] = _cnt(df["UniqueCarrier"].astype(str) + "@" + df["Origin"].astype(str), "co")
    X["cnt_cd"] = _cnt(df["UniqueCarrier"].astype(str) + "@" + df["Dest"].astype(str), "cd")
    X["n_carr_route"] = route.map(_route_carriers).fillna(1).to_numpy(dtype=float)
    # schedule position within the airport's day (train-fitted)
    X["origin_tod_cdf"] = _cdf(org_tod, df["Origin"].astype(str), X["tod"].to_numpy())
    arr = (X["tod"].to_numpy() + _ARR_TRAVEL_MIN + X["Distance"].to_numpy() / _ARR_SPEED_MPH * 60) % 1440
    X["dest_arr_cdf"] = _cdf(dst_arr, df["Dest"].astype(str), arr)
    # congestion share ratios (log-ratios of the counts above)
    X["share_oh"] = X["cnt_oh"] - X["cnt_origin"]
    X["share_dh"] = X["cnt_dh"] - X["cnt_dest"]
    X["share_co"] = X["cnt_co"] - X["cnt_origin"]
    X["share_cd"] = X["cnt_cd"] - X["cnt_dest"]
    X["share_route_hour"] = X["cnt_oh"] - X["cnt_dh"]
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model: diverse seed ensemble (depth 3/4/5, colsample 0.8) ------------------
t0 = time.time()
Xtr, Xe = prepare(train), prepare(evald)
ytr, ye = to_y(train), to_y(evald)
models = []
for depth, seed in zip([3, 4, 5] * 5, range(15)):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=depth,
        learning_rate=0.1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xe, ye)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in models]}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(ye, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
