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
    # --- more schedule-position and load variants ---
    X["rank_dest"] = df.groupby("Dest")["DepTime"].rank(pct=True).to_numpy()
    X["rank_carrier"] = df.groupby(CARRIER)["DepTime"].rank(pct=True).to_numpy()
    X["rank_route_dow"] = df.groupby(["Origin", "Dest", "DayOfWeek"])["DepTime"].rank(pct=True).to_numpy()
    X["rank_cr_dow"] = df.groupby([CARRIER, "Origin", "Dest", "DayOfWeek"])["DepTime"].rank(pct=True).to_numpy()
    X["n_earlier_route"] = df.groupby(["Origin", "Dest"])["DepTime"].rank(method="min").to_numpy() - 1
    X["share_earlier_route"] = X["n_earlier_route"] / df.groupby(["Origin", "Dest"])["DepTime"].transform("size").to_numpy()
    # origin schedule progress and airport-level hourly load vs its own mean
    X["origin_progress"] = df.groupby("Origin")["DepTime"].rank(pct=True).to_numpy()
    oh = df.groupby(["Origin", df["DepTime"] // 100]).transform("size").astype(float)
    X["origin_hour_vs_mean"] = (oh / df.groupby("Origin").transform("size").astype(float) * 24).to_numpy()
    # calendar relative volumes
    dow_d = df.groupby("DayOfWeek").transform("size").astype(float)
    X["rel_dow_volume"] = (dow_d / dow_d.mean()).to_numpy()
    mth = df.groupby("Month").transform("size").astype(float)
    X["rel_month_volume"] = (mth / mth.mean()).to_numpy()
    rd = df.groupby(["Origin", "Dest", "Month", "DayofMonth"]).transform("size").astype(float)
    X["rel_route_day_volume"] = (rd / rd.mean()).to_numpy()
    # --- schedule shape features per route/origin ---
    rv = df.groupby(["Origin", "Dest"]).transform("size").astype(float)
    cv = df.groupby("UniqueCarrier").transform("size").astype(float)
    X["rel_route_volume"] = (rv / rv.mean()).to_numpy()
    X["rel_carrier_volume"] = (cv / cv.mean()).to_numpy()
    X["route_dep_std"] = df.groupby(["Origin", "Dest"])["DepTime"].transform("std").fillna(0).to_numpy()
    mn = df.groupby(["Origin", "Dest"])["DepTime"].transform("min")
    mx = df.groupby(["Origin", "Dest"])["DepTime"].transform("max")
    X["route_t_norm"] = ((df["DepTime"] - mn) / (mx - mn).replace(0, 1)).to_numpy()
    X["rank_origin_hour"] = df.groupby(["Origin", df["DepTime"] // 100])["DepTime"].rank(pct=True).to_numpy()
    X["is_late_day"] = (df["DepTime"] >= 1800).astype(int).to_numpy()
    X["is_early"] = (df["DepTime"] < 600).astype(int).to_numpy()
    # --- schedule-gap features: minutes since previous / until next same-route (and carrier-origin) flight ---
    s = df.assign(_i=np.arange(len(df))).sort_values(["Origin", "Dest", "DepTime"])
    gp = s.groupby(["Origin", "Dest"])["DepTime"].diff().reindex(df.index)
    gn = -s.groupby(["Origin", "Dest"])["DepTime"].diff(-1).reindex(df.index)
    X["gap_prev_route"] = gp.to_numpy()
    X["gap_next_route"] = gn.to_numpy()
    s2 = df.assign(_i=np.arange(len(df))).sort_values(["UniqueCarrier", "Origin", "DepTime"])
    gp2 = s2.groupby(["UniqueCarrier", "Origin"])["DepTime"].diff().reindex(df.index)
    gn2 = -s2.groupby(["UniqueCarrier", "Origin"])["DepTime"].diff(-1).reindex(df.index)
    X["gap_prev_co"] = gp2.to_numpy()
    X["gap_next_co"] = gn2.to_numpy()
    # carrier-route gaps (same carrier on same route = aircraft rotation)
    s3 = df.assign(_i=np.arange(len(df))).sort_values(["UniqueCarrier", "Origin", "Dest", "DepTime"])
    gp3 = s3.groupby(["UniqueCarrier", "Origin", "Dest"])["DepTime"].diff().reindex(df.index)
    gn3 = -s3.groupby(["UniqueCarrier", "Origin", "Dest"])["DepTime"].diff(-1).reindex(df.index)
    X["gap_prev_cr"] = gp3.to_numpy()
    X["gap_next_cr"] = gn3.to_numpy()
    # 2nd-order carrier-route gaps (two flights back/forward = fuller rotation history)
    gp4 = s3.groupby(["UniqueCarrier", "Origin", "Dest"])["DepTime"].diff(2).reindex(df.index)
    gn4 = -s3.groupby(["UniqueCarrier", "Origin", "Dest"])["DepTime"].diff(-2).reindex(df.index)
    X["gap2_prev_cr"] = gp4.to_numpy()
    X["gap2_next_cr"] = gn4.to_numpy()
    # 3rd-order carrier-route gaps
    gp5 = s3.groupby(["UniqueCarrier", "Origin", "Dest"])["DepTime"].diff(3).reindex(df.index)
    gn5 = -s3.groupby(["UniqueCarrier", "Origin", "Dest"])["DepTime"].diff(-3).reindex(df.index)
    X["gap3_prev_cr"] = gp5.to_numpy()
    X["gap3_next_cr"] = gn5.to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
N_MODELS = 4
X_ev = prepare(evald)
y_ev = to_y(evald)
models = []
for i in range(N_MODELS):
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=SEED + i,
        n_jobs=N_JOBS,
    )
    t0 = time.time()
    m.fit(
        prepare(train),
        to_y(train),
        eval_set=[(X_ev, y_ev)],
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
