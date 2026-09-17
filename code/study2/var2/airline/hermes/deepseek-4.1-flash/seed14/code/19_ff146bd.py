"""XGBoost binary classifier + feature engineering. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Exp 3: ablations of the exp-2 feature set with a fixed-round model (no early stopping on a random split).
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

# ---------------------------------------------------------------- config -----
USE_TIME = True          # scheduled departure time-of-day features
USE_ROUTE = False        # Origin_Dest / carrier_origin high-cardinality categoricals
USE_LOG_DIST = True
USE_DENSITY = True       # airport-hour traffic share features
USE_CALENDAR = True      # month/day -> day-of-year, holiday windows
USE_INTERACT = True      # explicit products of continuous drivers
MAX_CARD = 5000
# -----------------------------------------------------------------------------

CAT_RAW = ["Month", "DayofMonth", "DayOfWeek"]


def _base(df: pd.DataFrame) -> pd.DataFrame:
    """Raw -> engineered frame. Pure per-row transforms only (no fitted statistics)."""
    X = pd.DataFrame(index=df.index)
    dist = pd.to_numeric(df["Distance"], errors="coerce")
    X["Distance"] = dist
    if USE_LOG_DIST:
        X["log_dist"] = np.log1p(dist)

    if USE_TIME:
        dep = pd.to_numeric(df["DepTime"], errors="coerce")
        dep = dep.where((dep >= 0) & (dep <= 2400))
        hour = np.floor(dep / 100.0).clip(0, 23)
        minute = dep - hour * 100
        minute = minute.where((minute >= 0) & (minute < 60))
        tod = hour + minute / 60.0
        X["dep_hour"] = hour
        X["tod"] = tod
        X["tod_sin"] = np.sin(2 * np.pi * tod / 24.0)
        X["tod_cos"] = np.cos(2 * np.pi * tod / 24.0)

    for c in CAT_RAW:
        X[c.lower()] = df[c].astype(str).replace({"nan": np.nan})
    X["carrier"] = df["UniqueCarrier"].astype(str).replace({"nan": np.nan})
    X["origin"] = df["Origin"].astype(str).replace({"nan": np.nan})
    X["dest"] = df["Dest"].astype(str).replace({"nan": np.nan})
    if USE_ROUTE:
        r = X["origin"].fillna("?") + "_" + X["dest"].fillna("?")
        X["route"] = r
        X["carrier_origin"] = X["carrier"].fillna("?") + "_" + X["origin"].fillna("?")
    if USE_DENSITY:
        # Traffic-structure features. Everything is a share of a count taken from the same frame, so the
        # values are comparable whether the frame holds 100k training rows or the 1M-row hidden holdout.
        hr = X["dep_hour"].astype("Int64").astype(str)
        o = X["origin"].fillna("?")
        d = X["dest"].fillna("?")
        c = X["carrier"].fillna("?")
        rt = o + "_" + d
        dow = X["dayofweek"].fillna("?")
        n = float(len(df))

        def _share(num: pd.Series, den: pd.Series) -> np.ndarray:
            return (num.map(num.value_counts()) / den.map(den.value_counts())).to_numpy(dtype=float)

        oh, dh, ch = o + "_" + hr, d + "_" + hr, c + "_" + hr
        rt_h = rt + "_" + hr
        cr = c + "_" + rt
        X["origin_hour_share"] = _share(oh, o)      # how busy this airport is at this hour
        X["dest_hour_share"] = _share(dh, d)
        X["hour_share"] = _share(hr, pd.Series("all", index=hr.index))
        X["origin_share"] = _share(o, pd.Series("all", index=hr.index))
        X["dest_share"] = _share(d, pd.Series("all", index=hr.index))
        X["route_hour_share"] = _share(rt_h, rt)
        X["carrier_hour_share"] = _share(ch, c)
        X["route_share"] = _share(rt, o)            # this route's share of the origin's departures
        X["carrier_route_share"] = _share(cr, c)    # this route's share of the carrier's schedule
        X["origin_hour_dow_share"] = _share(oh + "_" + dow, oh)
        X["dest_hour_dow_share"] = _share(dh + "_" + dow, dh)
        oc = o + "_" + c
        dc = d + "_" + c
        X["origin_carrier_share"] = _share(oc, o)   # hub dominance
        X["carrier_origin_share"] = _share(oc, c)
        X["dest_carrier_share"] = _share(dc, d)
        # fraction of that key's daily traffic scheduled at or before this hour (congestion build-up)
        X["origin_cum_hour_share"] = _cum_share(o, X["dep_hour"])
        X["dest_cum_hour_share"] = _cum_share(d, X["dep_hour"])
        X["carrier_cum_hour_share"] = _cum_share(c, X["dep_hour"])
        X["route_cum_hour_share"] = _cum_share(rt, X["dep_hour"])
        # finer (half-hour) resolution for the high-volume keys, plus carrier-at-origin rotation build-up
        hb = np.floor(pd.to_numeric(X["tod"], errors="coerce") * 2.0).astype("Int64").astype(str)
        ohb, dhb, chb = o + "_" + hb, d + "_" + hb, c + "_" + hb
        X["origin_halfhour_share"] = _share(ohb, o)
        X["dest_halfhour_share"] = _share(dhb, d)
        X["carrier_halfhour_share"] = _share(chb, c)
        coh = oc + "_" + hr
        X["carrier_origin_hour_share"] = _share(coh, oc)
        X["carrier_origin_cum_hour_share"] = _cum_share(oc, X["dep_hour"])
        # 15-minute resolution for carriers; global weekday-hour load; hour busy-ness rank within a key
        qb = np.floor(pd.to_numeric(X["tod"], errors="coerce") * 4.0).astype("Int64").astype(str)
        X["carrier_quarterhour_share"] = _share(c + "_" + qb, c)
        X["dow_hour_share"] = _share(dow + "_" + hr, dow)
        X["origin_hour_rank"] = _share_rank(o, hr)
        X["dest_hour_rank"] = _share_rank(d, hr)
        # arrival-side congestion: block time estimated from distance, then the destination's traffic at
        # that estimated arrival hour
        arr_tod = pd.to_numeric(X["tod"], errors="coerce") + pd.to_numeric(X["Distance"], errors="coerce") / 450.0
        ar = arr_tod % 24.0
        X["est_arr_tod"] = ar
        ah = np.floor(ar).astype("Int64")
        X["dest_arr_hour_share"] = _share(d + "_" + ah.astype(str), d)
        X["dest_arr_cum_hour_share"] = _cum_share(d, ah)
        # minute-within-the-hour scheduling pattern (airlines bank flights on :00/:15/:30/:45)
        mb = np.floor((pd.to_numeric(X["tod"], errors="coerce") * 60.0 % 60.0) / 15.0).astype("Int64").astype(str)
        X["origin_hour_minute_share"] = _share(oh + "_" + mb, oh)
        X["carrier_hour_minute_share"] = _share(ch + "_" + mb, ch)
        X["route_hour_rank"] = _share_rank(rt, hr)
        # minute-of-hour scheduling on the arrival/carrier side too
        X["dest_hour_minute_share"] = _share(dh + "_" + mb, dh)
        X["carrier_minute_share"] = _share(c + "_" + mb, c)
    if USE_CALENDAR:
        # Month / DayofMonth are ordinal codes (c-<n> = the real month / day), so calendar structure
        # (holiday travel windows, day-of-year seasonality) is recoverable.
        m = pd.to_numeric(df["Month"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
        dm = pd.to_numeric(df["DayofMonth"].astype(str).str.replace("c-", "", regex=False), errors="coerce")
        m_start = np.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=float)
        mm = np.clip(m.to_numpy(dtype=float, na_value=np.nan), 0, 12)
        doy = np.where(np.isnan(mm) | np.isnan(dm.to_numpy()), np.nan, m_start[np.nan_to_num(mm, nan=0).astype(int)] + dm.to_numpy())
        X["month_num"] = m
        X["day_num"] = dm
        X["doy"] = doy
        X["doy_sin"] = np.sin(2 * np.pi * doy / 365.0)
        X["doy_cos"] = np.cos(2 * np.pi * doy / 365.0)
        X["is_winter_holidays"] = (((m == 12) & (dm >= 18)) | ((m == 1) & (dm <= 3))).astype(float)
        X["is_thanksgiving"] = ((m == 11) & dm.between(20, 29)).astype(float)
        X["is_labor_day"] = ((m == 9) & (dm <= 5)).astype(float)
        X["is_memorial"] = ((m == 5) & (dm >= 25)).astype(float)
        X["is_july4"] = ((m == 7) & (dm <= 7)).astype(float)
    if USE_INTERACT:
        # explicit products of the strongest continuous drivers (trees approximate products poorly)
        X["tod_x_logdist"] = X["tod"] * X["log_dist"]
        X["hour_x_dist"] = X["dep_hour"] * X["Distance"]
        X["ohs_x_dhs"] = X["origin_hour_share"] * X["dest_hour_share"]
        X["ohs_x_chs"] = X["origin_hour_share"] * X["carrier_hour_share"]
        X["ohs_x_routes"] = X["origin_hour_share"] * X["route_share"]
        X["ohs_x_cum"] = X["origin_hour_share"] * X["origin_cum_hour_share"]
        X["hour_x_carrier_share"] = X["dep_hour"] * X["carrier_origin_share"]
    return X


def _cum_share(keys: pd.Series, hours: pd.Series) -> np.ndarray:
    t = pd.DataFrame({"k": keys.fillna("?"), "h": hours})
    cnt = t.groupby(["k", "h"], dropna=True).size()
    ratio = cnt.groupby(level=0).cumsum() / cnt.groupby(level=0).transform("sum")
    return t.set_index(["k", "h"]).index.map(ratio).to_numpy(dtype=float)


def _share_rank(keys: pd.Series, hours: pd.Series) -> np.ndarray:
    """Percentile of this hour's traffic share among the key's hours (how close to the key's peak)."""
    t = pd.DataFrame({"k": keys.fillna("?"), "h": hours})
    cnt = t.groupby(["k", "h"], dropna=True).size()
    share = cnt / cnt.groupby(level=0).transform("sum")
    rank = share.groupby(level=0).rank(pct=True)
    return t.set_index(["k", "h"]).index.map(rank).to_numpy(dtype=float)


NUMERIC = {"Distance", "log_dist", "dep_hour", "tod", "tod_sin", "tod_cos",
           "origin_hour_share", "dest_hour_share", "hour_share", "origin_share",
           "dest_share", "route_hour_share", "carrier_hour_share",
           "origin_hour_dow_share", "dest_hour_dow_share", "origin_carrier_share", "carrier_origin_share",
           "origin_cum_hour_share", "dest_cum_hour_share", "carrier_cum_hour_share",
           "route_share", "carrier_route_share", "dest_carrier_share", "route_cum_hour_share",
           "origin_halfhour_share", "dest_halfhour_share", "carrier_halfhour_share",
           "carrier_origin_hour_share", "carrier_origin_cum_hour_share",
           "carrier_quarterhour_share", "dow_hour_share", "origin_hour_rank", "dest_hour_rank",
           "est_arr_tod", "dest_arr_hour_share", "dest_arr_cum_hour_share",
           "origin_hour_minute_share", "carrier_hour_minute_share", "route_hour_rank",
           "dest_hour_minute_share", "carrier_minute_share",
           "month_num", "day_num", "doy", "doy_sin", "doy_cos",
           "is_winter_holidays", "is_thanksgiving", "is_labor_day", "is_memorial", "is_july4",
           "tod_x_logdist", "hour_x_dist", "ohs_x_dhs", "ohs_x_chs", "ohs_x_routes", "ohs_x_cum",
           "hour_x_carrier_share"}

_probe = _base(train)
ALL_COLS = list(_probe.columns)
cat_levels = {}
for c in ALL_COLS:
    if c in NUMERIC:
        continue
    vals = _probe[c].dropna().unique()
    if len(vals) <= MAX_CARD:
        cat_levels[c] = pd.Index(np.sort(vals))
feature_cols = [c for c in ALL_COLS if c in NUMERIC or c in cat_levels]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = _base(df)[feature_cols].copy()
    for c in feature_cols:
        if c in cat_levels:
            X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models (ensemble of XGBoost variants) ------------------------------------
BASE = dict(
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=4,
    n_jobs=N_JOBS,
)
PARAM_SETS = []
for _i, (_d, _lr, _ne, _mcw, _cs, _ss) in enumerate([
    (4, 0.05, 300, 10, 0.8, 0.8),
    (5, 0.05, 300, 10, 0.8, 0.8),
    (6, 0.04, 400, 20, 0.8, 0.8),
    (4, 0.03, 600, 20, 0.6, 0.8),
    (5, 0.03, 600, 30, 0.6, 0.8),
    (4, 0.05, 300, 10, 0.6, 0.8),
    (6, 0.05, 300, 30, 0.7, 0.8),
    (5, 0.04, 400, 10, 0.9, 0.8),
    (4, 0.04, 400, 10, 0.4, 0.7),
    (5, 0.04, 400, 20, 0.4, 0.9),
    (3, 0.05, 300, 5, 0.7, 0.7),
    (6, 0.03, 500, 40, 0.5, 0.9),
]):
    PARAM_SETS.append(dict(
        max_depth=_d, learning_rate=_lr, n_estimators=_ne, min_child_weight=_mcw,
        colsample_bytree=_cs, subsample=_ss, reg_lambda=2.0, random_state=SEED + _i,
    ))
# two leaf-wise members for structural diversity (same budget, different tree shape)
PARAM_SETS.append(dict(grow_policy="lossguide", max_depth=0, max_leaves=31, learning_rate=0.05,
                       n_estimators=300, min_child_weight=20, colsample_bytree=0.8, subsample=0.8,
                       reg_lambda=2.0, random_state=SEED + 20))
PARAM_SETS.append(dict(grow_policy="lossguide", max_depth=0, max_leaves=63, learning_rate=0.03,
                       n_estimators=500, min_child_weight=40, colsample_bytree=0.6, subsample=0.8,
                       reg_lambda=4.0, random_state=SEED + 21))

Xt = prepare(train)
yt = to_y(train)

models = []
t0 = time.time()
for ps in PARAM_SETS:
    m = xgb.XGBClassifier(**BASE, **ps)
    m.fit(Xt, yt, verbose=False)
    models.append(m)
print(f"Training time: {time.time() - t0:.1f}s  n_models={len(models)}  n_features={len(feature_cols)}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    dmat = xgb.DMatrix(prepare(df), enable_categorical=True)
    preds = np.mean([m.get_booster().predict(dmat) for m in models], axis=0)
    return preds


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
