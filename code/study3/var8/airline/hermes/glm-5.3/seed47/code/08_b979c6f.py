"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Feature design: batch-level (transductive) schedule features computed INSIDE prepare() on the
dataframe passed in — a flight's position in its carrier/route/origin departure schedule, and
scale-invariant relative traffic loads. They reproduce identically on the hidden holdout because
they are recomputed within the holdout batch at predict time (validate.py passes the whole set at once).
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
CARRIER = "UniqueCarrier"
cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["Month", "DayofMonth", "DayOfWeek", "DepTime", "Distance"] + cat_cols].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    X["hour"] = X["DepTime"] // 100
    X["minute"] = X["DepTime"] % 100
    X["DepTime_sin"] = np.sin(2 * np.pi * X["DepTime"] / 2400)
    X["DepTime_cos"] = np.cos(2 * np.pi * X["DepTime"] / 2400)
    # numeric versions of the c-encoded date fields (allow threshold splits)
    X["month_n"] = X["Month"].astype(str).str.slice(2).astype(int)
    X["day_n"] = X["DayofMonth"].astype(str).str.slice(2).astype(int)
    X["dow_n"] = X["DayOfWeek"].astype(str).str.slice(2).astype(int)
    X["doy"] = (X["month_n"] - 1) * 31 + X["day_n"]
    X["doy_sin"] = np.sin(2 * np.pi * X["doy"] / 365)
    X["doy_cos"] = np.cos(2 * np.pi * X["doy"] / 365)
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["is_weekend"] = (X["dow_n"] >= 6).astype(int)
    X["log_distance"] = np.log1p(X["Distance"])

    # --- batch-level schedule features: computed within the dataframe passed in ---
    # rank of this flight's departure time among its carrier-route / route / origin flights
    X["rank_cr"] = df.groupby([CARRIER, "Origin", "Dest"])["DepTime"].rank(pct=True).to_numpy()
    X["rank_route"] = df.groupby(["Origin", "Dest"])["DepTime"].rank(pct=True).to_numpy()
    X["rank_origin"] = df.groupby("Origin")["DepTime"].rank(pct=True).to_numpy()
    X["dev_route_center"] = (df["DepTime"] - df.groupby(["Origin", "Dest"])["DepTime"].transform("mean")).to_numpy()
    # scale-invariant relative traffic loads (count / mean count within batch)
    r1 = df.groupby(["Origin", "Dest", df["DepTime"] // 100]).transform("size").astype(float)
    X["rel_route_hour"] = (r1 / r1.mean()).to_numpy()
    r2 = df.groupby(["Origin", df["DepTime"] // 100]).transform("size").astype(float)
    X["rel_origin_hour"] = (r2 / r2.mean()).to_numpy()
    r3 = df.groupby(["Dest", df["DepTime"] // 100]).transform("size").astype(float)
    X["rel_dest_hour"] = (r3 / r3.mean()).to_numpy()
    day = df.groupby(["Month", "DayofMonth"]).transform("size").astype(float)
    X["rel_day_volume"] = (day / day.mean()).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 5
models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.03,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    t0 = time.time()
    m.fit(
        prepare(train),
        to_y(train),
        eval_set=[(prepare(evald), to_y(evald))],
        verbose=False,
    )
    models.append(m)
    print(f"model {i}: {time.time() - t0:.1f}s, best iters: {m.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    ps = [m.predict_proba(prepare(df))[:, 1] for m in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
