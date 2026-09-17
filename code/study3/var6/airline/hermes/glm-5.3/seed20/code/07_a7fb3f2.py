"""XGBoost binary classifier for airline delay prediction.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
     ALL feature engineering lives in prepare(df); statistics/categories are fit on train only.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42
N_BAGS = 5

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering -------------------------------------------------------
CAT_COLS = [
    "UniqueCarrier", "Origin", "Dest", "route",
    "hour_route", "hour_origin", "hour_dest", "carrier_hour", "hour_dow",
]


def _cnum(s):
    """c-7 -> 7 (robust to plain numbers too)."""
    return pd.to_numeric(s.astype(str).str.replace("c-", "", regex=False), errors="coerce")


def _base_features(df):
    X = pd.DataFrame(index=df.index)
    X["month"] = _cnum(df["Month"])
    X["day"] = _cnum(df["Month"])  # placeholder replaced below
    X["day"] = _cnum(df["DayofMonth"])
    X["dow"] = _cnum(df["DayOfWeek"])
    # scheduled departure: hhmm -> minutes of day (continuous)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = (dt // 100).clip(0, 26)
    mm = dt % 100
    minute_of_day = (hh * 60 + mm).astype(float)
    hour = (minute_of_day // 60).clip(0, 26)
    X["dep_time"] = dt
    X["minute_of_day"] = minute_of_day
    X["dep_hour"] = hour
    frac = minute_of_day / 1440.0
    X["dep_sin"] = np.sin(2 * np.pi * frac)
    X["dep_cos"] = np.cos(2 * np.pi * frac)
    X["distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["log_distance"] = np.log1p(X["distance"])
    origin = df["Origin"].astype(str)
    dest = df["Dest"].astype(str)
    carrier = df["UniqueCarrier"].astype(str)
    route = origin + "_" + dest
    X["route"] = route
    hs = hour.astype(int).astype(str)
    X["hour_route"] = hs + "_" + route
    X["hour_origin"] = hs + "_" + origin
    X["hour_dest"] = hs + "_" + dest
    X["carrier_hour"] = carrier + "_" + hs
    X["hour_dow"] = hs + "_" + X["dow"].astype("Int64").astype(str)
    X["UniqueCarrier"] = carrier
    X["Origin"] = origin
    X["Dest"] = dest
    return X


_base_train = _base_features(train)
cat_levels = {c: pd.Index(sorted(_base_train[c].unique())) for c in CAT_COLS}

# --- route-level distance statistics (fit on train only) -------------------------
_route_dist = _base_train.groupby("route", observed=True)["distance"].agg(["mean", "count", "std"])
_route_dist["std"] = _route_dist["std"].fillna(0)
route_dist_mean = _route_dist["mean"]
route_dist_cnt = _route_dist["count"]
route_dist_std = _route_dist["std"]
_global_dist_mean = float(_base_train["distance"].mean())

# --- within-route departure-time statistics (fit on train only) ------------------
_route_time = _base_train.groupby("route", observed=True)["minute_of_day"].agg(["mean", "std"])
_route_time["std"] = _route_time["std"].fillna(-1.0)
route_time_mean = _route_time["mean"]
route_time_std = _route_time["std"]


def _add_route_stats(X):
    rd_mean = X["route"].map(route_dist_mean).astype(float)
    X["route_dist_mean"] = rd_mean.fillna(_global_dist_mean).to_numpy()
    X["dist_minus_routemean"] = X["distance"] - X["route_dist_mean"]
    X["dist_ratio_routemean"] = X["distance"] / rd_mean
    rt_mean = X["route"].map(route_time_mean).astype(float)
    rt_std = X["route"].map(route_time_std).astype(float)
    ok = (rt_std > 0).to_numpy()
    z = np.zeros(len(X))
    z[ok] = ((X["minute_of_day"].to_numpy()[ok] - rt_mean.to_numpy()[ok]) / rt_std.to_numpy()[ok])
    X["route_time_z"] = z
    # within-(origin,date) departure order: later flights inherit the day's disruption
    okey = X["Origin"].astype(str) + "_" + X["month"].astype("Int64").astype(str) + "_" + X["day"].astype("Int64").astype(str)
    grp = X.groupby(okey, observed=True)["minute_of_day"]
    X["dep_order"] = grp.rank().to_numpy()
    X["dep_time_since_first"] = (X["minute_of_day"] - grp.transform("min")).to_numpy()
    return X


def prepare(df):
    X = _base_features(df)
    X = _add_route_stats(X)
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- bagged ensemble of XGBoost models ------------------------------------------
X_all = prepare(train)
y_all = to_y(train)

models = []
t0 = time.time()
for k in range(N_BAGS):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=SEED + k, stratify=y_all
    )
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=10,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=20,
        early_stopping_rounds=60,
        random_state=SEED + 100 * k,
        n_jobs=N_JOBS,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  iters={[m.best_iteration for m in models]}")


def predict_proba(df):
    Xp = prepare(df)
    p = np.zeros(len(Xp))
    for m in models:
        p += m.predict_proba(Xp)[:, 1]
    return p / len(models)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
