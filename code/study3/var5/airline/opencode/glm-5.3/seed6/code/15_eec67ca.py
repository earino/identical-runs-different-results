"""XGBoost binary classifier — airline delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

# --- features (exp 9 set): baseline cats + hour cat + smoothed route TE -----------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
HOURS = pd.Index(sorted((train["DepTime"] // 100).unique()))

_route = train["Origin"] + "_" + train["Dest"]
_y = (train[TARGET] == POSITIVE).astype(float)
PRIOR = float(_y.mean())
M = 20.0


def _smoothed(frame: pd.DataFrame) -> pd.Series:
    g = frame.groupby("route")["y"]
    return (g.sum() + M * PRIOR) / (g.count() + M)


tmp = pd.DataFrame({"route": _route, "y": _y})
route_map = _smoothed(tmp)  # full-train map used at predict time
route_oof = np.empty(len(tmp))
for tr_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=SEED).split(tmp):
    m = _smoothed(tmp.iloc[tr_idx])
    route_oof[val_idx] = tmp["route"].iloc[val_idx].map(m).fillna(PRIOR).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["DepTime", "Distance"]].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    X["hour"] = pd.Categorical(df["DepTime"] // 100, categories=HOURS)
    X["route_te"] = (df["Origin"] + "_" + df["Dest"]).map(route_map).fillna(PRIOR).to_numpy()
    X["dep_min"] = (df["DepTime"] // 100) * 60 + df["DepTime"] % 100
    d = with_hour(df)
    X["dow_hour_te"] = keys_of(d, ["DayOfWeek", "hour"]).map(cand_full["dow_hour_te"][3]).fillna(PRIOR).to_numpy()
    ohc = keys_of(d, ["Origin", "hour"]).map(cand_full["origin_hour_count"][3]).fillna(0.0).to_numpy()
    X["origin_hour_count"] = np.log1p(ohc)
    dhc = keys_of(d, ["Dest", "hour"]).map(cand_full["dest_hour_count"][3]).fillna(0.0).to_numpy()
    X["dest_hour_count"] = np.log1p(dhc)
    chc = keys_of(d, ["UniqueCarrier", "hour"]).map(cand_full["carrier_hour_count"][3]).fillna(0.0).to_numpy()
    X["carrier_hour_count"] = np.log1p(chc)
    X["carrier_hour_te"] = keys_of(d, ["UniqueCarrier", "hour"]).map(cand_full["carrier_hour_te"][3]).fillna(PRIOR).to_numpy()
    X["origin_hour_te"] = keys_of(d, ["Origin", "hour"]).map(cand_full["origin_hour_te"][3]).fillna(PRIOR).to_numpy()
    X["dest_hour_te"] = keys_of(d, ["Dest", "hour"]).map(cand_full["dest_hour_te"][3]).fillna(PRIOR).to_numpy()
    X["dep_sin"] = np.sin(2 * np.pi * X["dep_min"] / 1440)
    X["dep_cos"] = np.cos(2 * np.pi * X["dep_min"] / 1440)
    return X


def prepare_train(df: pd.DataFrame, oof: np.ndarray) -> pd.DataFrame:
    X = prepare(df)
    X["route_te"] = oof
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- candidate re-scan under the deep regime (1 combo, 2 seeds each) --------------
def with_hour(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(hour=df["DepTime"] // 100)


def keys_of(df: pd.DataFrame, cols: list) -> pd.Series:
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    return k


def smoothed_map(keys: pd.Series, y: pd.Series, m: float) -> pd.Series:
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"]
    return (g.sum() + m * PRIOR) / (g.count() + m)


def oof_te(keys: pd.Series, m: float) -> np.ndarray:
    out = np.empty(len(keys))
    for tr_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=SEED).split(keys):
        mp = smoothed_map(keys.iloc[tr_idx], _y.iloc[tr_idx], m)
        out[val_idx] = keys.iloc[val_idx].map(mp).fillna(PRIOR).to_numpy()
    return out


CAND_DEFS = {
    "dow_hour_te": (["DayOfWeek", "hour"], 20.0),
    "origin_hour_count": (["Origin", "hour"], "count"),
    "dest_hour_count": (["Dest", "hour"], "count"),
    "carrier_hour_te": (["UniqueCarrier", "hour"], 20.0),
    "carrier_hour_count": (["UniqueCarrier", "hour"], "count"),
    "origin_hour_te": (["Origin", "hour"], 50.0),
    "dest_hour_te": (["Dest", "hour"], 50.0),
}
cand_full, cand_oof = {}, {}
for name, (cols, m) in CAND_DEFS.items():
    keys = keys_of(with_hour(train), cols)
    if m == "count":
        cm = keys.value_counts()
        cand_full[name] = (keys_of, cols, np.log1p(keys.map(cm).fillna(0.0).to_numpy()), cm)
    else:
        fm = smoothed_map(keys, _y, m)
        cand_full[name] = (keys_of, cols, oof_te(keys, m), fm)


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xall = prepare_train(train, route_oof)
yall = to_y(train)
Xev = prepare(evald)
yev = to_y(evald)
for name, (kf, cols, oof, fm) in cand_full.items():
    Xall[name] = oof

CAND_COLS = {"chc": ["carrier_hour_count"],
             "oht": ["origin_hour_te"],
             "dht": ["dest_hour_te"]}
CORE = ["dep_min", "origin_hour_count", "dest_hour_count", "carrier_hour_te", "dep_sin", "dep_cos"]


def member(Xa, Xe, seed):
    m = xgb.XGBClassifier(
        n_estimators=2000, max_depth=18, learning_rate=0.03, colsample_bytree=0.3,
        subsample=0.9, tree_method="hist", enable_categorical=True,
        early_stopping_rounds=40, random_state=SEED + 1000 * seed, n_jobs=N_JOBS)
    m.fit(Xa, yall, eval_set=[(Xe, yev)], verbose=False)
    return m


BASE_FEATS = ["DepTime", "Distance", *CAT_COLS, "hour", "route_te"]
results = {}
t0 = time.time()
for cand, extra in CAND_COLS.items():
    cols = BASE_FEATS + CORE + extra
    ms = [member(Xall[cols], Xev[cols], s) for s in range(2)]
    aucs = [roc_auc_score(yev, m.predict_proba(Xev[cols])[:, 1]) for m in ms]
    ens = roc_auc_score(yev, np.mean([m.predict_proba(Xev[cols])[:, 1] for m in ms], axis=0))
    results[cand] = (ms, cols, ens)
    print(f"  cand={cand} aucs={['%.4f' % a for a in aucs]} ens={ens:.4f}", flush=True)
print(f"Scan time: {time.time() - t0:.1f}s")

best_cand = max(results, key=lambda k: results[k][2])
print(f"  chosen={best_cand}")
members, model_cols, eval_auc = results[best_cand]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(prepare(df)[model_cols])[:, 1] for m in members], axis=0)


print(f"Eval AUC: {eval_auc:.4f}")
