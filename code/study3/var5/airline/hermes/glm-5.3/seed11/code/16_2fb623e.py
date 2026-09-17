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

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- volume/congestion statistics, fit on train only --------------------------
NB = 48  # 30-minute buckets in a day


def _mins(dep: pd.Series) -> pd.Series:
    ht = (dep // 100).clip(0, 24)
    return ht * 60 + dep % 100


_tr_mins = _mins(train["DepTime"])
_tr_keys = pd.DataFrame({
    "Origin": train["Origin"].astype(str),
    "Dest": train["Dest"].astype(str),
    "Carrier": train["UniqueCarrier"].astype(str),
    "H": (_tr_mins // 60).astype(str),
    "D": train["DayOfWeek"].astype(str),
    "M": train["Month"].astype(str),
    "B": ((_tr_mins // 30).astype(int) % NB),
})

CNT_UNSEEN = 1.0
cnt_maps = {}


def _fit_cnt(name: str, keys: pd.Series) -> None:
    cnt_maps[name] = keys.value_counts().astype("float32")


_fit_cnt("Carrier", _tr_keys["Carrier"])
_fit_cnt("Origin", _tr_keys["Origin"])
_fit_cnt("Dest", _tr_keys["Dest"])
_fit_cnt("Route", _tr_keys["Origin"] + "_" + _tr_keys["Dest"])
_fit_cnt("OriginHour", _tr_keys["Origin"] + "_" + _tr_keys["H"])
_fit_cnt("DestHour", _tr_keys["Dest"] + "_" + _tr_keys["H"])
_fit_cnt("OriginDOW", _tr_keys["Origin"] + "_" + _tr_keys["D"])
_fit_cnt("OriginHourDOW", _tr_keys["Origin"] + "_" + _tr_keys["H"] + "_" + _tr_keys["D"])
_fit_cnt("CarrierHour", _tr_keys["Carrier"] + "_" + _tr_keys["H"])
_fit_cnt("OriginHM", _tr_keys["Origin"] + "_" + (_tr_mins // 30).astype(str))
_fit_cnt("OriginMonthHour", _tr_keys["Origin"] + "_" + _tr_keys["M"] + "_" + _tr_keys["H"])
_fit_cnt("RouteHour", _tr_keys["Origin"] + "_" + _tr_keys["Dest"] + "_" + _tr_keys["H"])
_fit_cnt("CarrierRoute", _tr_keys["Carrier"] + "_" + _tr_keys["Origin"] + "_" + _tr_keys["Dest"])
_fit_cnt("DestHourDOW", _tr_keys["Dest"] + "_" + _tr_keys["H"] + "_" + _tr_keys["D"])
_fit_cnt("CarrierDOW", _tr_keys["Carrier"] + "_" + _tr_keys["D"])
_fit_cnt("CarrierOriginHour", _tr_keys["Carrier"] + "_" + _tr_keys["Origin"] + "_" + _tr_keys["H"])
_fit_cnt("MonthOriginHour", _tr_keys["M"] + "_" + _tr_keys["Origin"] + "_" + _tr_keys["H"])

# rolling-window queue sizes: departures at origin / arrivals at dest within +-2 buckets (±1h)
_airports = sorted(set(_tr_keys["Origin"]) | set(_tr_keys["Dest"]))
_ap_ix = {a: i for i, a in enumerate(_airports)}


def _bucket_matrix(ap: pd.Series, buckets: pd.Series) -> np.ndarray:
    """airports x 48 count matrix from train."""
    m = np.zeros((len(_airports), NB), dtype="float32")
    keys = pd.DataFrame({"a": ap.map(_ap_ix).to_numpy(), "b": buckets.to_numpy()})
    cnt = keys.groupby(["a", "b"]).size()
    m[cnt.index.get_level_values(0), cnt.index.get_level_values(1)] = cnt.values
    return m


_dep_mat = _bucket_matrix(_tr_keys["Origin"], _tr_keys["B"])
_arr_bucket = (((_tr_mins + train["Distance"] / 450.0 * 60.0) // 30).astype(int) % NB)
_arr_mat = _bucket_matrix(_tr_keys["Dest"], _arr_bucket)


def _roll(mat: np.ndarray, half: int) -> np.ndarray:
    out = np.zeros_like(mat)
    for off in range(-half, half + 1):
        out += np.roll(mat, off, axis=1)
    return out


_dep_roll30 = _roll(_dep_mat, 1)
_dep_roll1h = _roll(_dep_mat, 2)
_dep_roll2h = _roll(_dep_mat, 4)
_arr_roll30 = _roll(_arr_mat, 1)
_arr_roll1h = _roll(_arr_mat, 2)
_arr_roll2h = _roll(_arr_mat, 4)
_dep_fine = _dep_mat.copy()  # exact bucket count
_ap_daily = (np.tile(_dep_mat.sum(axis=1, keepdims=True), (1, NB)) / 365.0).astype("float32")


def parse_c(s: pd.Series) -> pd.Series:
    """'c-4' -> 4 (float, NaN if malformed)."""
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    # numeric, cyclical versions of the calendar columns
    for c, period in (("Month", 12), ("DayofMonth", 31), ("DayOfWeek", 7)):
        n = parse_c(X[c])
        X[c] = n
        X[c + "_sin"] = np.sin(2 * np.pi * n / period)
        X[c + "_cos"] = np.cos(2 * np.pi * n / period)
    # departure time: hours + minutes + cyclical
    ht = (X["DepTime"] // 100).clip(0, 24)
    mt = X["DepTime"] % 100
    mins = ht * 60 + mt
    X["DepHour"] = ht
    X["DepMin"] = mt
    X["DepMinutes"] = mins
    X["DepMinutes_sin"] = np.sin(2 * np.pi * mins / 1440)
    X["DepMinutes_cos"] = np.cos(2 * np.pi * mins / 1440)
    # frequency encoding (unseen levels -> 1)
    org = X["Origin"].astype(str)
    dst = X["Dest"].astype(str)
    car = X["UniqueCarrier"].astype(str)
    h = (mins // 60).astype(str)
    d = X["DayOfWeek"].astype(str)
    m = X["Month"].astype(str)
    X["RouteCnt"] = (org + "_" + dst).map(cnt_maps["Route"]).fillna(CNT_UNSEEN).astype("float32")
    X["CarrierCnt"] = car.map(cnt_maps["Carrier"]).fillna(CNT_UNSEEN).astype("float32")
    X["OriginCnt"] = org.map(cnt_maps["Origin"]).fillna(CNT_UNSEEN).astype("float32")
    X["DestCnt"] = dst.map(cnt_maps["Dest"]).fillna(CNT_UNSEEN).astype("float32")
    # congestion features
    X["OriginHourCnt"] = (org + "_" + h).map(cnt_maps["OriginHour"]).fillna(CNT_UNSEEN).astype("float32")
    X["DestHourCnt"] = (dst + "_" + h).map(cnt_maps["DestHour"]).fillna(CNT_UNSEEN).astype("float32")
    X["OriginDOWCnt"] = (org + "_" + d).map(cnt_maps["OriginDOW"]).fillna(CNT_UNSEEN).astype("float32")
    X["OriginHourDOWCnt"] = (org + "_" + h + "_" + d).map(cnt_maps["OriginHourDOW"]).fillna(CNT_UNSEEN).astype("float32")
    X["CarrierHourCnt"] = (car + "_" + h).map(cnt_maps["CarrierHour"]).fillna(CNT_UNSEEN).astype("float32")
    X["OriginHMCnt"] = (org + "_" + (mins // 30).astype(str)).map(cnt_maps["OriginHM"]).fillna(CNT_UNSEEN).astype("float32")
    X["OriginMonthHourCnt"] = (org + "_" + m + "_" + h).map(cnt_maps["OriginMonthHour"]).fillna(CNT_UNSEEN).astype("float32")
    X["RouteHourCnt"] = (org + "_" + dst + "_" + h).map(cnt_maps["RouteHour"]).fillna(CNT_UNSEEN).astype("float32")
    X["CarrierRouteCnt"] = (car + "_" + org + "_" + dst).map(cnt_maps["CarrierRoute"]).fillna(CNT_UNSEEN).astype("float32")
    X["DestHourDOWCnt"] = (dst + "_" + h + "_" + d).map(cnt_maps["DestHourDOW"]).fillna(CNT_UNSEEN).astype("float32")
    X["CarrierDOWCnt"] = (car + "_" + d).map(cnt_maps["CarrierDOW"]).fillna(CNT_UNSEEN).astype("float32")
    X["CarrierOriginHourCnt"] = (car + "_" + org + "_" + h).map(cnt_maps["CarrierOriginHour"]).fillna(CNT_UNSEEN).astype("float32")
    X["MonthOriginHourCnt"] = (m + "_" + org + "_" + h).map(cnt_maps["MonthOriginHour"]).fillna(CNT_UNSEEN).astype("float32")
    # queue sizes: departures at origin / arrivals at dest within +-1h (train-fitted matrices)
    b = (mins // 30).astype(int) % NB
    org_ix = org.map(_ap_ix).fillna(-1).astype(int).to_numpy()
    dst_ix = dst.map(_ap_ix).fillna(-1).astype(int).to_numpy()
    safe_o = np.where(org_ix >= 0, org_ix, 0)
    safe_d = np.where(dst_ix >= 0, dst_ix, 0)
    X["DepQueue30"] = np.where(org_ix >= 0, _dep_roll30[safe_o, b.to_numpy()], 0.0)
    X["DepQueue1h"] = np.where(org_ix >= 0, _dep_roll1h[safe_o, b.to_numpy()], 0.0)
    X["DepQueue2h"] = np.where(org_ix >= 0, _dep_roll2h[safe_o, b.to_numpy()], 0.0)
    X["DepBucket"] = np.where(org_ix >= 0, _dep_fine[safe_o, b.to_numpy()], 0.0)
    arr_b = (((mins + X["Distance"] / 450.0 * 60.0) // 30).astype(int) % NB)
    X["ArrQueue30"] = np.where(dst_ix >= 0, _arr_roll30[safe_d, arr_b.to_numpy()], 0.0)
    X["ArrQueue1h"] = np.where(dst_ix >= 0, _arr_roll1h[safe_d, arr_b.to_numpy()], 0.0)
    X["ArrQueue2h"] = np.where(dst_ix >= 0, _arr_roll2h[safe_d, arr_b.to_numpy()], 0.0)
    X["AirportDaily"] = np.where(org_ix >= 0, _ap_daily[safe_o, 0], 0.0)
    # relative congestion: queue vs the airport's typical daily volume
    X["QueueRel"] = X["DepQueue1h"] / (X["AirportDaily"] + 1.0)
    X["QueueRel2h"] = X["DepQueue2h"] / (X["AirportDaily"] + 1.0)
    # queue x time-of-day interactions
    X["Queue_x_Hour"] = X["DepQueue1h"] * ht.to_numpy()
    X["QueueRel_x_Hour"] = X["QueueRel"] * ht.to_numpy()
    X["ArrQueue_x_Hour"] = X["ArrQueue1h"] * ht.to_numpy()
    # volume x time / distance interactions
    X["OriginHourCnt_x_Hour"] = X["OriginHourCnt"] * ht.to_numpy()
    X["DestHourCnt_x_Hour"] = X["DestHourCnt"] * ht.to_numpy()
    X["Dist_x_Hour"] = X["Distance"] * ht.to_numpy()
    X["DistPerMin_x_Hour"] = (X["Distance"] / mins.clip(lower=30)) * ht.to_numpy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: 3 XGBs, colsample diversity, mcw=1 ------------------------------
SPECS = [
    dict(max_depth=10, subsample=0.8, colsample_bytree=0.6, random_state=1),
    dict(max_depth=8, subsample=0.7, colsample_bytree=0.5, random_state=2),
    dict(max_depth=12, subsample=0.9, colsample_bytree=0.6, random_state=3),
]
models = [
    xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        min_child_weight=1,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=True,
        n_jobs=N_JOBS,
        **spec,
    )
    for spec in SPECS
]

t0 = time.time()
Xtr, ytr = prepare(train), to_y(train)
for m in models:
    m.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
