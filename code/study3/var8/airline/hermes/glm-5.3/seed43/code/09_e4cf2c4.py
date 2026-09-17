"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Recipe: smoothed target encoding of categorical keys (carrier, origin, dest, route,
route x hour, origin x hour, dest x hour) computed out-of-fold for the training matrix and
from full-train maps at predict time; numeric date/time features; XGBoost d8/mcw10.
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
SMOOTH = 50
y_int = (train[TARGET] == POSITIVE).astype(int)
PRIOR = float(y_int.mean())


def _hour(df):
    return (np.floor(df["DepTime"].astype(float) / 100) % 24).astype(int)


def _route_key(df):
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


def _keys(df):
    """All TE keys, built identically for train / eval / holdout."""
    h = _hour(df).astype(str)
    r = _route_key(df)
    dist_b = (df["Distance"].astype(float) // 250).astype(int).astype(str)
    return {
        "carrier": df["UniqueCarrier"].astype(str),
        "origin": df["Origin"].astype(str),
        "dest": df["Dest"].astype(str),
        "route": r,
        "route_hour": r + "_" + h,
        "origin_hour": df["Origin"].astype(str) + "_" + h,
        "dest_hour": df["Dest"].astype(str) + "_" + h,
        "carrier_dist": df["UniqueCarrier"].astype(str) + "_d" + dist_b,
    }


TRAIN_KEYS = _keys(train)


# full-data target-encoding maps (train only) used at predict time
def _te_map(key_series):
    st = pd.DataFrame({"k": key_series, "y": y_int}).groupby("k")["y"].agg(["sum", "count"])
    return (st["sum"] + SMOOTH * PRIOR) / (st["count"] + SMOOTH)


te_maps = {name: _te_map(s) for name, s in TRAIN_KEYS.items()}

# per-key training counts (confidence for the TE values)
te_counts = {}
for name, s in TRAIN_KEYS.items():
    te_counts[name] = s.value_counts()

# --- train-only aggregates (no target involved) --------------------------------
_route_key_tr = _route_key(train)
route_dist = train.groupby(_route_key_tr)["Distance"].mean()
route_cnt = train.groupby(_route_key_tr).size()
origin_cnt = train.groupby(train["Origin"].astype(str)).size()
dest_cnt = train.groupby(train["Dest"].astype(str)).size()
carrier_cnt = train.groupby(train["UniqueCarrier"].astype(str)).size()
origin_dist = train.groupby(train["Origin"].astype(str))["Distance"].mean()
dest_dist = train.groupby(train["Dest"].astype(str))["Distance"].mean()
origin_div = train.groupby(train["Origin"].astype(str))["Dest"].nunique()
dest_div = train.groupby(train["Dest"].astype(str))["Origin"].nunique()
route_dep = train.groupby(_route_key_tr)["DepTime"].mean()
origin_dep = train.groupby(train["Origin"].astype(str))["DepTime"].mean()
DIST_MEAN = float(train["Distance"].mean())
DEPT_MEAN = float(train["DepTime"].mean())


def _cnum(s):
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    X["month_num"] = _cnum(df["Month"])
    X["dom_num"] = _cnum(df["DayofMonth"])
    X["dow_num"] = _cnum(df["DayOfWeek"])
    dt = df["DepTime"].astype(float)
    hour = np.floor(dt / 100) % 24
    X["dep_time"] = dt.to_numpy()
    X["minute_of_day"] = (hour * 60 + (dt - np.floor(dt / 100) * 100)).to_numpy()
    X["hour_sin"] = np.sin(2 * np.pi * X["minute_of_day"] / 1440)
    X["hour_cos"] = np.cos(2 * np.pi * X["minute_of_day"] / 1440)
    X["distance"] = df["Distance"].astype(float).to_numpy()
    keys = _keys(df)
    for name, series in keys.items():
        X[name + "_te"] = series.map(te_maps[name]).fillna(PRIOR).to_numpy()
    # train-only aggregates
    r = _route_key(df)
    o = df["Origin"].astype(str)
    d = df["Dest"].astype(str)
    X["route_avg_dist"] = r.map(route_dist).fillna(DIST_MEAN).to_numpy()
    X["route_cnt"] = np.log1p(r.map(route_cnt).fillna(0).to_numpy())
    X["origin_cnt"] = np.log1p(o.map(origin_cnt).fillna(0).to_numpy())
    X["dest_cnt"] = np.log1p(d.map(dest_cnt).fillna(0).to_numpy())
    X["carrier_cnt"] = np.log1p(df["UniqueCarrier"].astype(str).map(carrier_cnt).fillna(0).to_numpy())
    X["origin_avg_dist"] = o.map(origin_dist).fillna(DIST_MEAN).to_numpy()
    X["dest_avg_dist"] = d.map(dest_dist).fillna(DIST_MEAN).to_numpy()
    X["origin_div"] = o.map(origin_div).fillna(0).to_numpy()
    X["dest_div"] = d.map(dest_div).fillna(0).to_numpy()
    X["route_avg_dep"] = r.map(route_dep).fillna(DEPT_MEAN).to_numpy()
    X["origin_avg_dep"] = o.map(origin_dep).fillna(DEPT_MEAN).to_numpy()
    X["dist_vs_route_avg"] = X["distance"].to_numpy() - X["route_avg_dist"].to_numpy()
    # TE confidence: training-set support for the hour-interaction keys
    for k in ["route", "route_hour", "origin_hour", "dest_hour"]:
        X[k + "_cnt"] = np.log1p(keys[k].map(te_counts[k]).fillna(0).to_numpy())
    return X


def prepare_train() -> pd.DataFrame:
    # training matrix with OUT-OF-FOLD target encoding (no leakage)
    X = prepare(train)
    rng = np.random.RandomState(SEED)
    fold = rng.randint(0, 5, len(train))
    yv = y_int.to_numpy()
    for name, series in TRAIN_KEYS.items():
        col = np.full(len(train), PRIOR)
        kv = series.to_numpy()
        for f in range(5):
            m = fold == f
            st = pd.DataFrame({"k": kv[~m], "y": yv[~m]}).groupby("k")["y"].agg(["sum", "count"])
            enc = (st["sum"] + SMOOTH * PRIOR) / (st["count"] + SMOOTH)
            col[m] = pd.Series(kv[m]).map(enc).fillna(PRIOR).to_numpy()
        X[name + "_te"] = col
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=480,
    max_depth=8,
    min_child_weight=10,
    learning_rate=0.05,
    colsample_bytree=0.7,
    tree_method="hist",
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(prepare_train(), y_int.to_numpy())
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
