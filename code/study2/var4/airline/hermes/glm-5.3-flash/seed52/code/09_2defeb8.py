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
SEED = 42

TE_COLS = ["UniqueCarrier", "Origin", "Dest", "Route"]
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
CAT_LEVELS = {}  # fitted on training data only; unseen levels map to NaN
TE_MAPS = {}  # category -> smoothed target mean, fitted on training rows only
TE_PRIOR = 0.0
TE_ALPHA = 40.0  # smoothing strength for target encoding


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + ">" + df["Dest"].astype(str)


def _reset_encoders() -> None:
    CAT_LEVELS.clear()
    TE_MAPS.clear()
    global TE_PRIOR
    TE_PRIOR = 0.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    dt = pd.to_numeric(df["DepTime"], errors="coerce")
    hh = np.floor(dt / 100.0)
    mm = dt - hh * 100
    # scheduled times >= 2400 (after-midnight roll) wrap into the early morning
    hh = np.where(hh >= 24, hh - 24, hh)
    hour = hh
    mod = hour * 60 + mm  # minutes of day, cyclic in [0, 1440)
    X["DepHour"] = hour
    X["DepMinute"] = mm
    X["DepMinOfDay"] = mod
    X["DepSinH"] = np.sin(2 * np.pi * hour / 24.0)
    X["DepCosH"] = np.cos(2 * np.pi * hour / 24.0)
    X["DepSinD"] = np.sin(2 * np.pi * mod / 1440.0)
    X["DepCosD"] = np.cos(2 * np.pi * mod / 1440.0)
    for c, name in (("Month", "Month"), ("DayofMonth", "DayOfMonth"), ("DayOfWeek", "DayOfWeek")):
        X[name] = pd.to_numeric(df[c].astype(str).str.lstrip("c-"), errors="coerce")
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    X["LogDistance"] = np.log1p(dist)
    route = _route(df)
    for c in CAT_COLS:
        levels = CAT_LEVELS.get(c)
        if levels is None:  # first call is on the training frame: fit
            levels = pd.Index(sorted(df[c].dropna().astype(str).unique()))
            CAT_LEVELS[c] = levels
        X[c] = pd.Categorical(df[c].astype("object").where(df[c].notna(), None).astype(str),
                              categories=levels)
    # target encoding: category -> smoothed mean of the labels of the frame prepare()
    # was first called on (always training rows); later frames only look up.
    if not TE_MAPS:
        global TE_PRIOR
        y = to_y(df)
        TE_PRIOR = float(y.mean())
        groups = {"UniqueCarrier": X["UniqueCarrier"], "Origin": X["Origin"], "Dest": X["Dest"],
                  "Route": route}
        for c, g in groups.items():
            stats = pd.Series(y, index=df.index).groupby(g.astype("object").to_numpy()).agg(["mean", "count"])
            enc = (stats["mean"] * stats["count"] + TE_PRIOR * TE_ALPHA) / (stats["count"] + TE_ALPHA)
            TE_MAPS[c] = enc
    for c, g in (("UniqueCarrier", X["UniqueCarrier"]), ("Origin", X["Origin"]),
                 ("Dest", X["Dest"]), ("Route", route)):
        X[c + "_te"] = g.astype("object").to_numpy()
        X[c + "_te"] = pd.Series(X[c + "_te"]).map(TE_MAPS[c]).astype(float).fillna(TE_PRIOR).to_numpy()
    # distance-bucket x hour TE: long-haul delay curve peaks later in the day
    dbucket = np.digitize(np.asarray(X["Distance"], dtype=float), [350.0, 750.0, 1500.0])
    hour_f = np.floor(np.asarray(X["DepMinOfDay"], dtype=float) / 60.0)
    dh_key = pd.Series(dbucket).astype(str) + "|" + pd.Series(hour_f).astype(str)
    if "DistHour" not in TE_MAPS:
        stats = pd.Series(y, index=df.index).groupby(dh_key.to_numpy()).agg(["mean", "count"])
        TE_MAPS["DistHour"] = (stats["mean"] * stats["count"] + TE_PRIOR * 60.0) / (stats["count"] + 60.0)
    X["DistHour_te"] = pd.Series(dh_key.to_numpy()).map(TE_MAPS["DistHour"]).astype(float) \
        .fillna(TE_PRIOR).to_numpy()
    # carrier x hour TE: regional vs major carriers have different delay curves
    ch_key = pd.Series(X["UniqueCarrier"].astype("object").to_numpy()).astype(str) + "|" \
        + pd.Series(hour_f).astype(str)
    if "CarrierHour" not in TE_MAPS:
        stats = pd.Series(y, index=df.index).groupby(ch_key.to_numpy()).agg(["mean", "count"])
        TE_MAPS["CarrierHour"] = (stats["mean"] * stats["count"] + TE_PRIOR * 60.0) / (stats["count"] + 60.0)
    X["CarrierHour_te"] = pd.Series(ch_key.to_numpy()).map(TE_MAPS["CarrierHour"]).astype(float) \
        .fillna(TE_PRIOR).to_numpy()
    X = X.drop(columns=["Route"], errors="ignore")
    return X


def make_model(n_estimators: int, learning_rate: float, seed: int = SEED) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        early_stopping_rounds=50,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        n_jobs=N_JOBS,
    )


train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# stage 1: early stopping on a chronological split (last 20% of 2005 rows) — mimics the
# 2005 -> 2006 shift better than a random split and picks the tree count
t0 = time.time()
_reset_encoders()
tr = train.iloc[:80000]
va = train.iloc[80000:]
es_model = make_model(4000, 0.03)
es_model.fit(prepare(tr), to_y(tr), eval_set=[(prepare(va), to_y(va))], verbose=False)
best_ntree = int(es_model.best_iteration) + 1
print(f"Best iteration: {best_ntree}")

# stage 2: refit on ALL training rows; encoders re-fit on the full training frame.
# Bag of 4 models with different seeds/subsample streams, averaged.
_reset_encoders()
X_full = prepare(train)
y_full = to_y(train)
models = []
for k in range(6):
    mk = make_model(int(best_ntree * 1.15), 0.03, seed=SEED + 101 * k)
    mk.set_params(early_stopping_rounds=None, random_state=SEED + 101 * k)
    mk.fit(X_full, y_full)
    models.append(mk)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xp = prepare(df)
    return np.mean([m.predict_proba(Xp)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
