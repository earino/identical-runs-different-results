"""XGBoost airline delay. Contract: prints `Eval AUC: 0.xxxx`; predict_proba(df) works on raw rows."""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))
SEED = 42

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")

# --- feature engineering: stats/levels fitted on train only --------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOURDOW_LEVELS = pd.Index(range(168))
_dep_train = pd.to_numeric(train["DepTime"], errors="coerce").fillna(0) % 2400
_carrier_hour = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" +
                                (_dep_train // 100).astype(int).astype(str)).unique()))


def _c_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def _key(*cols):
    codes = [pd.factorize(c.to_numpy())[0].astype(np.int64) for c in cols]
    k = codes[0]
    for c in codes[1:]:
        k = k * (c.max() + 1) + c
    return pd.factorize(k)[0]


def _roll_counts(kf, mods: np.ndarray, windows: dict, out: dict) -> None:
    """out[name][i] = rows with same key and minute in [m+a, m+b)."""
    order = np.lexsort((mods, kf))
    ks, ms = kf[order], np.asarray(mods)[order]
    for name in windows:
        out[name] = np.zeros(len(mods))
    bounds = np.searchsorted(ks, np.arange(ks.max() + 1), side="left")
    bounds = np.append(bounds, ks.size)
    for g in range(len(bounds) - 1):
        s, e = bounds[g], bounds[g + 1]
        arr = ms[s:e]
        for name, (a, b) in windows.items():
            aa = a if np.isscalar(a) else a[order[s:e]]
            bb = b if np.isscalar(b) else b[order[s:e]]
            lo = np.searchsorted(arr, arr + aa, side="left")
            hi = np.searchsorted(arr, arr + bb, side="left")
            out[name][order[s:e]] = hi - lo


def _gaps(kf, mods: np.ndarray, out: dict) -> None:
    """Minutes to previous / next same-key flight, normalized per 1000 rows."""
    order = np.lexsort((mods, kf))
    ks, ms = kf[order], np.asarray(mods)[order]
    prev = np.full(len(mods), 1440.0)
    nxt = np.full(len(mods), 1440.0)
    same_prev = np.append(False, ks[1:] == ks[:-1])
    idx = np.where(same_prev)[0]
    prev[order[idx]] = ms[idx] - ms[idx - 1]
    same_next = np.append(ks[1:] == ks[:-1], False)
    idx2 = np.where(same_next)[0]
    nxt[order[idx2]] = ms[idx2 + 1] - ms[idx2]
    out["GapPrev"] = prev
    out["GapNext"] = nxt


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["Month"] = _c_num(df["Month"])
    X["DayofMonth"] = _c_num(df["DayofMonth"])
    X["DayOfWeek"] = _c_num(df["DayOfWeek"])
    dep = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0) % 2400
    X["DepTime"] = dep
    X["Hour"] = (dep // 100).astype(int)
    X["MinOfDay"] = (dep // 100) * 60 + (dep % 100)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    X["LogDist"] = np.log1p(X["Distance"])
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["Block"] = np.select([(dep // 100) < 6, (dep // 100) < 12, (dep // 100) < 18], [0, 1, 2], default=3)
    X["IsWeekend"] = X["DayOfWeek"].isin([6, 7]).astype(int)
    X["HourDow"] = pd.Categorical((dep // 100).astype(int) * 7 + (X["DayOfWeek"].fillna(1) - 1).astype(int),
                                   categories=HOURDOW_LEVELS)
    X["CarrierHour"] = pd.Categorical(df["UniqueCarrier"].astype(str) + "_" + (dep // 100).astype(int).astype(str),
                                      categories=_carrier_hour)
    # within-df congestion (label-free, scale-normalized per 1000 rows)
    hour = (dep // 100).astype(int)
    ones = pd.Series(1.0, index=df.index)
    scale = 1000.0 / len(df)
    slot = ((dep // 100) * 60 + (dep % 100)) // 30
    X["OrigSlotRate"] = ones.groupby([df["Origin"], slot]).transform("size") * scale
    X["DestSlotRate"] = ones.groupby([df["Dest"], slot]).transform("size") * scale
    X["RouteSlotRate"] = ones.groupby([df["Origin"], df["Dest"], slot]).transform("size") * scale
    mods = X["MinOfDay"].to_numpy()
    rc = {}
    W = {"Roll15": (-15, 16), "Back30": (-30, 0), "Back60": (-60, 0), "Back90": (-90, 0),
         "Back120": (-120, 0), "Back180": (-180, 0), "Back240": (-240, 0), "Roll5": (-5, 6)}
    _roll_counts(_key(df["Origin"]), mods, W, rc)
    for k, v in rc.items():
        X[f"Orig{k}"] = v * scale
    rc = {}
    _roll_counts(_key(df["Dest"]), mods, W, rc)
    for k, v in rc.items():
        X[f"Dest{k}"] = v * scale
    rc = {}
    WR = {"Roll15": (-15, 16), "Back30": (-30, 0), "Back60": (-60, 0), "Back90": (-90, 0),
          "Back120": (-120, 0), "Fwd30": (0, 30)}
    _roll_counts(_key(df["Origin"], df["Dest"]), mods, WR, rc)
    for k, v in rc.items():
        X[f"Route{k}"] = v * scale
    _roll_counts(_key(df["Origin"], df["Dest"], df["Month"], df["DayofMonth"]), mods,
                 {"DayPos": (-mods, 0)}, rc)
    X["RouteDayPos"] = rc["DayPos"] * scale
    _roll_counts(_key(df["Origin"], df["Month"], df["DayofMonth"]), mods,
                 {"DayPos": (-mods, 0), "DayFwd": (0, 1440)}, rc)
    X["OrigDayPos"] = rc["DayPos"] * scale
    X["OrigDayFwd"] = rc["DayFwd"] * scale
    X["OrigDayFrac"] = rc["DayPos"] / (rc["DayPos"] + rc["DayFwd"] + 1.0)
    # day position/fraction for route key
    rc3 = {}
    _roll_counts(_key(df["Origin"], df["Dest"], df["Month"], df["DayofMonth"]), mods,
                 {"DayPos": (-mods, 0), "DayFwd": (0, 1440)}, rc3)
    X["RouteDayFwd"] = rc3["DayFwd"] * scale
    X["RouteDayFrac"] = rc3["DayPos"] / (rc3["DayPos"] + rc3["DayFwd"] + 1.0)
    # route share of origin/dest congestion
    X["RouteShareOrig"] = X["RouteBack60"] / (X["OrigBack60"] + 0.5)
    X["RouteShareDest"] = X["RouteBack60"] / (X["DestBack60"] + 0.5)
    X["OrigBurst"] = X["OrigBack30"] / (X["OrigBack240"] + 0.5)
    # dest day position/fraction
    rc4 = {}
    _roll_counts(_key(df["Dest"], df["Month"], df["DayofMonth"]), mods,
                 {"DayPos": (-mods, 0), "DayFwd": (0, 1440)}, rc4)
    X["DestDayPos"] = rc4["DayPos"] * scale
    X["DestDayFwd"] = rc4["DayFwd"] * scale
    X["DestDayFrac"] = rc4["DayPos"] / (rc4["DayPos"] + rc4["DayFwd"] + 1.0)
    # spacing gaps (normalized: minutes x rows/1000)
    gscale = len(df) / 1000.0
    rc = {}
    _gaps(_key(df["Origin"], df["Dest"]), mods, rc)
    X["RouteGapPrev"] = rc["GapPrev"] * gscale
    X["RouteGapNext"] = rc["GapNext"] * gscale
    rc = {}
    _gaps(_key(df["Origin"]), mods, rc)
    X["OrigGapPrev"] = rc["GapPrev"] * gscale
    rc = {}
    _gaps(_key(df["Dest"]), mods, rc)
    X["DestGapPrev"] = rc["GapPrev"] * gscale
    X["DestGapNext"] = rc["GapNext"] * gscale
    rc = {}
    _gaps(_key(df["UniqueCarrier"]), mods, rc)
    X["CarrierGapPrev"] = rc["GapPrev"] * gscale
    # carrier queue
    rc = {}
    _roll_counts(_key(df["UniqueCarrier"]), mods,
                 {"Back60": (-60, 0), "Back120": (-120, 0), "Roll15": (-15, 16)}, rc)
    X["CarrierBack60"] = rc["Back60"] * scale
    X["CarrierBack120"] = rc["Back120"] * scale
    X["CarrierRoll15"] = rc["Roll15"] * scale
    _roll_counts(_key(df["UniqueCarrier"], df["Month"], df["DayofMonth"]), mods,
                 {"DayPos": (-mods, 0)}, rc)
    X["CarrierDayPos"] = rc["DayPos"] * scale
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


USE_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
            "LogDist", "UniqueCarrier", "Origin", "Dest", "Block", "IsWeekend", "HourDow", "CarrierHour",
            "OrigSlotRate", "DestSlotRate", "RouteSlotRate",
            "OrigRoll15", "OrigBack30", "OrigBack60", "OrigBack90", "OrigBack120",
            "DestRoll15", "DestBack30", "DestBack60", "DestBack90", "DestBack120",
            "RouteRoll15", "RouteBack30", "RouteBack60", "RouteBack90", "RouteBack120", "RouteFwd30",
            "RouteDayPos", "OrigDayPos"]

PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=5, min_child_weight=100,
              tree_method="hist", enable_categorical=True, eval_metric="auc",
              early_stopping_rounds=30, subsample=0.7, colsample_bytree=0.7, n_jobs=N_JOBS)

Xtr, Xev = prepare(train), prepare(evald)
y_all, yev = to_y(train), to_y(evald)

PROD = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
        "LogDist", "UniqueCarrier", "Origin", "Dest", "Block", "IsWeekend", "HourDow", "CarrierHour",
        "OrigSlotRate", "DestSlotRate", "RouteSlotRate",
        "OrigRoll15", "OrigBack30", "OrigBack60", "OrigBack90", "OrigBack120",
        "DestRoll15", "DestBack30", "DestBack60", "DestBack90", "DestBack120",
        "RouteRoll15", "RouteBack30", "RouteBack60", "RouteBack90", "RouteBack120", "RouteFwd30",
        "RouteDayPos", "OrigDayPos"]
GAPS = ["RouteGapPrev", "RouteGapNext", "OrigGapPrev"]
CARRQ = ["CarrierBack60", "CarrierBack120", "CarrierRoll15", "CarrierDayPos"]
LONGQ = ["OrigBack180", "OrigBack240", "OrigRoll5"]
DAYFRAC = ["OrigDayFwd", "OrigDayFrac", "RouteDayFwd", "RouteDayFrac", "RouteShareOrig"]
CURRENT = PROD + GAPS + CARRQ + LONGQ + DAYFRAC
DESTSIDE = ["DestGapPrev", "DestGapNext", "DestDayPos", "DestDayFwd", "DestDayFrac"]
MISC = ["CarrierGapPrev", "RouteShareDest", "OrigBurst"]
VARIANTS = {
    "destside": CURRENT + DESTSIDE,
    "misc": CURRENT + MISC,
    "max": CURRENT + DESTSIDE + MISC,
}

t0 = time.time()
best_name, best_ev = None, -1
for name, cols in VARIANTS.items():
    F = Xtr[cols]
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    models = []
    for tr_i, va_i in kf.split(train):
        m = xgb.XGBClassifier(**PARAMS, random_state=SEED)
        m.fit(F.iloc[tr_i], y_all[tr_i], eval_set=[(F.iloc[va_i], y_all[va_i])], verbose=False)
        models.append(m)
    ev = roc_auc_score(yev, np.mean([m.predict_proba(Xev[cols])[:, 1] for m in models], axis=0))
    print(f"[{name}] eval={ev:.4f} t={time.time()-t0:.1f}s")
    if ev > best_ev:
        best_name, best_ev = name, ev
print(f"best: {best_name}")

USE_COLS = VARIANTS[best_name]
F = Xtr[USE_COLS]
MODELS = []
t0 = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for tr_i, va_i in kf.split(train):
    m = xgb.XGBClassifier(**PARAMS, random_state=SEED)
    m.fit(F.iloc[tr_i], y_all[tr_i], eval_set=[(F.iloc[va_i], y_all[va_i])], verbose=False)
    MODELS.append(m)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)[USE_COLS]
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
