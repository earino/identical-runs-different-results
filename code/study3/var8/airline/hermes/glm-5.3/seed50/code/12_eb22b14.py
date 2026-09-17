"""XGBoost binary classifier: airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
obj_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]
            and (pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c]))]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}


def _num(s):  # 'c-7' -> 7
    return pd.to_numeric(s.astype(str).str.replace(r"^[a-zA-Z]-", "", regex=True), errors="coerce")


# --- smoothed target statistics fit on training data only ---------------------
_y = (train[TARGET] == POSITIVE).astype(float)
_global_mean = float(_y.mean())
SMOOTH_M = 20.0


def _stats_by(key_series: pd.Series) -> dict:
    g = pd.DataFrame({"k": key_series, "y": _y}).groupby("k")["y"].agg(["mean", "count"])
    return {k: (float(r["mean"]), float(r["count"])) for k, r in g.iterrows()}


carrier_stats = _stats_by(train["UniqueCarrier"])
origin_stats = _stats_by(train["Origin"])
dest_stats = _stats_by(train["Dest"])

# interaction categorical levels (fit on train only)
_tr_mins = (((train["DepTime"].astype(float) // 100) % 24) * 60 + (train["DepTime"].astype(float) % 100))
_tr_hh = ((train["DepTime"].astype(float) // 100) % 24)
car_block_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + np.floor(_tr_mins / 120.0).astype(int).astype(str)).unique()))
car_hour_levels = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" + _tr_hh.astype(int).astype(str)).unique()))

# smoothed rates for interaction keys (fit on train only)
car_hour_stats = _stats_by(train["UniqueCarrier"].astype(str) + "_" + _tr_hh.astype(int).astype(str))
car_blk_stats = _stats_by(train["UniqueCarrier"].astype(str) + "_" + np.floor(_tr_mins / 120.0).astype(int).astype(str))
org_blk_stats = _stats_by(train["Origin"].astype(str) + "_" + np.floor(_tr_mins / 120.0).astype(int).astype(str))
# distance-band x carrier and hour-band x distance rates
_dist_bin = pd.cut(train["Distance"], bins=[0, 250, 500, 1000, 2000, 6000], labels=["1", "2", "3", "4", "5"]).astype(str)
car_dist_stats = _stats_by(train["UniqueCarrier"].astype(str) + "_" + _dist_bin)
hh_dist_stats = _stats_by(_tr_hh.astype(int).astype(str) + "_" + _dist_bin)
# finer time bins x carrier
_tr_b30 = np.floor(_tr_mins / 30.0).astype(int).astype(str)
car_b30_stats = _stats_by(train["UniqueCarrier"].astype(str) + "_" + _tr_b30)
# origin coarsened x block rate
_top_origins = train["Origin"].value_counts()
major_origins = set(_top_origins[_top_origins >= 500].index.tolist())
_org_coarse = train["Origin"].where(train["Origin"].isin(major_origins), "OTHER")
org_coarse_blk_stats = _stats_by(_org_coarse + "_" + np.floor(_tr_mins / 120.0).astype(int).astype(str))
org_coarse_hour_stats = _stats_by(_org_coarse + "_" + _tr_hh.astype(int).astype(str))

# congestion counts: flights per (origin, hour) and per (origin, block) in train
org_hour_cnt = train.groupby([train["Origin"], _tr_hh.astype(int)]).size().to_dict()
org_blk_cnt = train.groupby([train["Origin"], np.floor(_tr_mins / 120.0).astype(int)]).size().to_dict()
dest_hour_cnt = train.groupby([train["Dest"], _tr_hh.astype(int)]).size().to_dict()
tot_per_hour = train.groupby(_tr_hh.astype(int)).size().to_dict()
avg_org_hour = float(np.mean(list(org_hour_cnt.values())))


def _smoothed(mapping: dict, key) -> float:
    if key in mapping:
        mean, n = mapping[key]
        return (mean * n + _global_mean * SMOOTH_M) / (n + SMOOTH_M)
    return _global_mean


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = pd.DataFrame(index=df.index)
    for c in cat_cols:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])  # unseen levels -> NaN
    X["MonthN"] = _num(df["Month"])
    X["DayN"] = _num(df["DayofMonth"])
    X["DowN"] = _num(df["DayOfWeek"])
    dep = df["DepTime"].astype(float)
    hh = (dep // 100) % 24
    mm = dep % 100
    mins = hh * 60 + mm
    X["DepHour"] = hh
    X["DepMinute"] = mm
    X["DepMins"] = mins
    X["DepSin"] = np.sin(2 * np.pi * mins / 1440.0)
    X["DepCos"] = np.cos(2 * np.pi * mins / 1440.0)
    X["Distance"] = df["Distance"].astype(float)
    X["CarrierRate"] = df["UniqueCarrier"].map(lambda k: _smoothed(carrier_stats, k))
    X["OriginRate"] = df["Origin"].map(lambda k: _smoothed(origin_stats, k))
    X["DestRate"] = df["Dest"].map(lambda k: _smoothed(dest_stats, k))
    # time-of-day blocks and finer bins
    blk = pd.Series(np.floor(mins / 120.0), index=df.index)  # 2-hour block (0..11)
    X["DepBlock2h"] = blk
    b15 = pd.Series(np.floor(mins / 15.0), index=df.index)  # 15-min bins (0..95)
    X["DepBin15"] = b15
    # interactions with distance
    X["Hour_x_Dist"] = hh * df["Distance"].astype(float) / 1000.0
    X["LogDist"] = np.log1p(df["Distance"].astype(float))
    # carrier x time-of-day risk
    car_block = (df["UniqueCarrier"].astype(str) + "_" + np.floor(mins / 120.0).astype(int).astype(str))
    car_hour = (df["UniqueCarrier"].astype(str) + "_" + hh.astype(int).astype(str))
    X["Car_x_Block"] = pd.Categorical(car_block, categories=car_block_levels)
    X["Car_x_Hour"] = pd.Categorical(car_hour, categories=car_hour_levels)
    # smoothed interaction rates as dense numerics
    X["CarHourRate"] = car_hour.map(lambda k: _smoothed(car_hour_stats, k))
    X["CarBlkRate"] = car_block.map(lambda k: _smoothed(car_blk_stats, k))
    org_blk = df["Origin"].astype(str) + "_" + np.floor(mins / 120.0).astype(int).astype(str)
    X["OrgBlkRate"] = org_blk.map(lambda k: _smoothed(org_blk_stats, k))
    # congestion features (train-fitted counts)
    _okey = list(zip(df["Origin"], hh.astype(int)))
    X["OrgHourCnt"] = [org_hour_cnt.get(k, avg_org_hour / 24) for k in _okey]
    _obkey = list(zip(df["Origin"], np.floor(mins / 120.0).astype(int)))
    X["OrgBlkCnt"] = [org_blk_cnt.get(k, avg_org_hour / 2) for k in _obkey]
    _dkey = list(zip(df["Dest"], hh.astype(int)))
    X["DestHourCnt"] = [dest_hour_cnt.get(k, avg_org_hour / 24) for k in _dkey]
    X["TotHourCnt"] = [tot_per_hour.get(int(h), 0) for h in hh]
    # distance-band interactions
    _db = pd.cut(df["Distance"], bins=[0, 250, 500, 1000, 2000, 6000], labels=["1", "2", "3", "4", "5"]).astype(str)
    X["CarDistRate"] = (df["UniqueCarrier"].astype(str) + "_" + _db).map(lambda k: _smoothed(car_dist_stats, k))
    X["HhDistRate"] = (hh.astype(int).astype(str) + "_" + _db).map(lambda k: _smoothed(hh_dist_stats, k))
    # finer time x carrier, coarse-origin x time
    X["CarB30Rate"] = (df["UniqueCarrier"].astype(str) + "_" + np.floor(mins / 30.0).astype(int).astype(str)).map(
        lambda k: _smoothed(car_b30_stats, k))
    _oc = df["Origin"].where(df["Origin"].isin(major_origins), "OTHER")
    X["OrgCBlkRate"] = (_oc + "_" + np.floor(mins / 120.0).astype(int).astype(str)).map(
        lambda k: _smoothed(org_coarse_blk_stats, k))
    X["OrgCHourRate"] = (_oc + "_" + hh.astype(int).astype(str)).map(
        lambda k: _smoothed(org_coarse_hour_stats, k))
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
SEEDS = [42, 7, 2024]
models = []
t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
Xev, yev = prepare(evald), to_y(evald)
for seed in SEEDS:
    m = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.0,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    m.fit(Xtr, ytr, eval_set=[(Xev, yev)], verbose=False)
    models.append(m)
    print(f"seed {seed}: best_iter={m.best_iteration}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xq = prepare(df)
    ps = np.mean([m.predict_proba(Xq)[:, 1] for m in models], axis=0)
    return ps


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
