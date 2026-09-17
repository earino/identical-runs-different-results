"""XGBoost airline delay. Contract: prints `Eval AUC: 0.xxxx`; predict_proba(df) works on raw rows."""
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

# --- feature engineering: stats/levels fitted on train only --------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOURDOW_LEVELS = pd.Index(range(168))
_dep_train = pd.to_numeric(train["DepTime"], errors="coerce").fillna(0) % 2400
_carrier_hour = pd.Index(sorted((train["UniqueCarrier"].astype(str) + "_" +
                                (_dep_train // 100).astype(int).astype(str)).unique()))


def _c_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def _roll_counts(key_cols: list, mods: np.ndarray, windows: list, out: dict) -> None:
    """out[name][i] = rows sharing same key(s) with minute in [m+a, m+b)."""
    if len(key_cols) == 1:
        kf = pd.factorize(key_cols[0].to_numpy())[0]
    else:
        kf = pd.factorize([tuple(r) for r in zip(*[k.to_numpy() for k in key_cols])])[0]
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
            lo = np.searchsorted(arr, arr + a, side="left")
            hi = np.searchsorted(arr, arr + b, side="left")
            out[name][order[s:e]] = hi - lo


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
    # within-df congestion (label-free; rolling window counts, scale-normalized)
    hour = (dep // 100).astype(int)
    ones = pd.Series(1.0, index=df.index)
    scale = 1000.0 / len(df)
    slot = ((dep // 100) * 60 + (dep % 100)) // 30
    X["OrigHourRate"] = ones.groupby([df["Origin"], hour]).transform("size") * scale
    X["DestHourRate"] = ones.groupby([df["Dest"], hour]).transform("size") * scale
    X["RouteHourRate"] = ones.groupby([df["Origin"], df["Dest"], hour]).transform("size") * scale
    X["RouteSlotRate"] = ones.groupby([df["Origin"], df["Dest"], slot]).transform("size") * scale
    X["OrigSlotRate"] = ones.groupby([df["Origin"], slot]).transform("size") * scale
    X["DestSlotRate"] = ones.groupby([df["Dest"], slot]).transform("size") * scale
    mods = (dep // 100) * 60 + (dep % 100)
    rc = {}
    W = {"Roll15": (-15, 16), "Back30": (-30, 0), "Back60": (-60, 0)}
    _roll_counts([df["Origin"]], mods, W, rc)
    for k, v in rc.items():
        X[f"Orig{k}"] = v * scale
    rc = {}
    _roll_counts([df["Dest"]], mods, W, rc)
    for k, v in rc.items():
        X[f"Dest{k}"] = v * scale
    rc = {}
    WR = {"Roll15": (-15, 16), "Back30": (-30, 0), "Back60": (-60, 0), "Fwd30": (0, 30)}
    _roll_counts([df["Origin"], df["Dest"]], mods, WR, rc)
    for k, v in rc.items():
        X[f"Route{k}"] = v * scale
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


USE_COLS = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
            "LogDist", "UniqueCarrier", "Origin", "Dest", "Block", "IsWeekend", "HourDow", "CarrierHour"]

PARAMS = dict(n_estimators=3000, learning_rate=0.03, max_depth=5, min_child_weight=100,
              tree_method="hist", enable_categorical=True, eval_metric="auc",
              early_stopping_rounds=30, subsample=0.7, colsample_bytree=0.7, n_jobs=N_JOBS)

Xtr, Xev = prepare(train), prepare(evald)
y_all, yev = to_y(train), to_y(evald)

BASE = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "Hour", "MinOfDay", "Distance",
        "LogDist", "UniqueCarrier", "Origin", "Dest", "Block", "IsWeekend", "HourDow", "CarrierHour"]
CONG = ["OrigSlotRate", "DestSlotRate", "RouteSlotRate"]
ROLL = ["OrigRoll15", "OrigBack30", "OrigBack60", "DestRoll15", "DestBack30", "DestBack60",
        "RouteRoll15", "RouteBack30", "RouteBack60", "RouteFwd30"]
VARIANTS = {
    "ref": BASE + CONG + ["OrigRoll15", "DestRoll15", "RouteRoll15"],
    "asym": BASE + CONG + ["OrigRoll15", "OrigBack30", "DestRoll15", "DestBack30",
                           "RouteRoll15", "RouteBack30", "RouteFwd30"],
    "full": BASE + CONG + ROLL,
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
MODELS = []
t0 = time.time()
F = Xtr[USE_COLS]
kf = KFold(n_splits=10, shuffle=True, random_state=0)
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
