"""Airline delay XGBoost classifier — experiment: + target encoding (5-fold OOF, smoothed).

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
from sklearn.model_selection import KFold

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
TE_COLS = ["te_carrier", "te_origin", "te_dest", "te_route", "te_month", "te_dow", "te_hour",
           "te_origin_hour", "te_dest_hour", "te_carrier_hour"]
NUM_COLS = ["DepTime", "hour", "minute_of_day", "hour_sin", "hour_cos",
            "month_sin", "month_cos", "dow_sin", "dow_cos",
            "dom_num", "dow_num", "month_num", "Distance", "log_dist",
            "log_n_route"] + TE_COLS

cat_levels = {}
for c in CAT_COLS:
    cat_levels[c] = pd.Index(sorted(train[c].dropna().unique()))

# target encoder keys: (colname -> smoothing m); route gets heavier smoothing
TE_SPECS = {
    "te_carrier": ("UniqueCarrier", 20),
    "te_origin": ("Origin", 100),
    "te_dest": ("Dest", 100),
    "te_route": ("route", 200),
    "te_month": ("Month", 20),
    "te_dow": ("DayOfWeek", 20),
    "te_hour": ("hour", 20),
    "te_origin_hour": ("origin_hour", 100),
    "te_dest_hour": ("dest_hour", 100),
    "te_carrier_hour": ("carrier_hour", 100),
}


def _key_df(df: pd.DataFrame) -> pd.DataFrame:
    k = pd.DataFrame(index=df.index)
    hh = (df["DepTime"].fillna(0).astype(int) // 100) % 24
    k["UniqueCarrier"] = df["UniqueCarrier"]
    k["Origin"] = df["Origin"]
    k["Dest"] = df["Dest"]
    k["route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    k["Month"] = df["Month"]
    k["DayOfWeek"] = df["DayOfWeek"]
    k["hour"] = hh
    k["origin_hour"] = df["Origin"].astype(str) + "_" + hh.astype(str)
    k["dest_hour"] = df["Dest"].astype(str) + "_" + hh.astype(str)
    k["carrier_hour"] = df["UniqueCarrier"].astype(str) + "_" + hh.astype(str)
    return k


def _fit_maps(keys: pd.DataFrame, y: np.ndarray, prior: float):
    maps = {}
    for name, (col, m) in TE_SPECS.items():
        g = pd.DataFrame({"k": keys[col].values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
        maps[name] = ((g["sum"] + m * prior) / (g["count"] + m)).to_dict()
    return maps


def _apply_maps(keys: pd.DataFrame, maps: dict, prior: float) -> pd.DataFrame:
    out = pd.DataFrame(index=keys.index)
    for name, (col, m) in TE_SPECS.items():
        out[name] = keys[col].map(maps[name]).fillna(prior).astype(float)
    return out


y_all = to_y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_all.mean())
keys_all = _key_df(train)

# full-train maps (used by prepare() for any unseen data)
FULL_MAPS = _fit_maps(keys_all, y_all, prior)

# 5-fold OOF maps for the training rows themselves (no in-fold leakage)
OOF_TE = pd.DataFrame(index=train.index, columns=TE_COLS, dtype=float)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for tr_idx, va_idx in kf.split(train):
    maps_f = _fit_maps(keys_all.iloc[tr_idx], y_all[tr_idx], prior)
    OOF_TE.iloc[va_idx] = _apply_maps(keys_all.iloc[va_idx], maps_f, prior).values

# counts fitted on train only
ROUTE_COUNTS = keys_all["route"].value_counts().to_dict()

DIST_BINS = [-1, 200, 400, 700, 1000, 1500, 2500, 10**6]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    dt = df["DepTime"].fillna(0).astype(int)
    hh = (dt // 100) % 24
    mm = dt % 100
    hour = hh.astype(float)
    mod = hour * 60.0 + mm
    dom = df["DayofMonth"].astype(str).str.replace("c-", "", regex=False).astype(float)
    dow = df["DayOfWeek"].astype(str).str.replace("c-", "", regex=False).astype(float)
    mon = df["Month"].astype(str).str.replace("c-", "", regex=False).astype(float)

    X["DepTime"] = df["DepTime"].astype(float)
    X["hour"] = hour
    X["minute_of_day"] = mod
    X["hour_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    X["hour_cos"] = np.cos(2 * np.pi * mod / 1440.0)
    X["month_sin"] = np.sin(2 * np.pi * (mon - 1) / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * (mon - 1) / 12.0)
    X["dow_sin"] = np.sin(2 * np.pi * (dow - 1) / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * (dow - 1) / 7.0)
    X["dom_num"] = dom
    X["dow_num"] = dow
    X["month_num"] = mon
    X["Distance"] = df["Distance"].astype(float)
    X["log_dist"] = np.log1p(df["Distance"].clip(lower=0))

    keys = _key_df(df)
    te = _apply_maps(keys, FULL_MAPS, prior)
    for c in TE_COLS:
        X[c] = te[c]
    X["log_n_route"] = np.log1p(keys["route"].map(ROUTE_COUNTS).fillna(0).astype(float))

    raw_route = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    for c, src in [("Month", df["Month"]), ("DayofMonth", df["DayofMonth"]),
                   ("DayOfWeek", df["DayOfWeek"]), ("UniqueCarrier", df["UniqueCarrier"]),
                   ("Origin", df["Origin"]), ("Dest", df["Dest"])]:
        X[c] = pd.Categorical(src, categories=cat_levels[c])

    return X[CAT_COLS + NUM_COLS]


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_EST = 700
SEEDS = [42, 43, 44]

t0 = time.time()
Xtr = prepare(train)
# training rows get OOF (leak-free) target-encoding values
for c in TE_COLS:
    Xtr[c] = OOF_TE[c].values
ytr = to_y(train)

models = []
for sd in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=N_EST,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=1.0,
        tree_method="hist",
        enable_categorical=True,
        random_state=sd,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s ({len(SEEDS)} models)")


# diagnostic: eval-AUC learning curve of the averaged ensemble
booster = models[0].get_booster()
Xe = prepare(evald)
for k in [200, 300, 450, 600, 700]:
    ps = [b.predict(xgb.DMatrix(Xe, enable_categorical=True), iteration_range=(0, k))
          for b in [m.get_booster() for m in models]]
    pk = np.mean(ps, axis=0)
    print(f"iter {k:4d}: eval_auc={roc_auc_score(to_y(evald), pk):.4f}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    ps = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
