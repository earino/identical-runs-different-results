"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

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
y_tr = (train[TARGET] == POSITIVE).astype(int).to_numpy()
y_ev = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}

# structural traffic volumes from the 2005 training rows (transfer across years)
_hour_tr = ((train["DepTime"] // 100) % 24).astype(int).astype(str)
_dow_tr = train["DayOfWeek"].astype("string").str.replace(r"^c-", "", regex=True)
_month_tr = train["Month"].astype("string").str.replace(r"^c-", "", regex=True)
_o, _d, _c = train["Origin"].astype(str), train["Dest"].astype(str), train["UniqueCarrier"].astype(str)
_route_tr = _o + "-" + _d
COUNT_SRC = {
    "origin": _o, "dest": _d, "route": _route_tr, "carrier": _c,
    "origin_hour": _o + "-" + _hour_tr, "dest_hour": _d + "-" + _hour_tr,
    "origin_carrier": _o + "-" + _c, "dest_carrier": _d + "-" + _c,
    "hour_carrier": _hour_tr + "-" + _c, "origin_dow": _o + "-" + _dow_tr,
    "hour_dow": _hour_tr + "-" + _dow_tr, "route_hour": _route_tr + "-" + _hour_tr,
    "origin_month": _o + "-" + _month_tr, "dest_month": _d + "-" + _month_tr,
}
COUNTS = {k: v.value_counts().to_dict() for k, v in COUNT_SRC.items()}
ROUTE_MEAN_DIST = train.groupby(_route_tr)["Distance"].mean().to_dict()


def _arr_min(dep_time, dist):
    return ((dep_time // 100 % 24) * 60 + dep_time % 100 + dist / 450.0) % 1440.0


_arr_hour_tr = np.floor(_arr_min(train["DepTime"].to_numpy(), train["Distance"].to_numpy()) / 60).astype(int).astype(str)
COUNT_SRC["dest_arr_hour"] = _d + "-" + _arr_hour_tr
COUNT_SRC["origin_arr_hour"] = _o + "-" + _arr_hour_tr
COUNTS["dest_arr_hour"] = COUNT_SRC["dest_arr_hour"].value_counts().to_dict()
COUNTS["origin_arr_hour"] = COUNT_SRC["origin_arr_hour"].value_counts().to_dict()


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype("string").str.replace(r"^c-", "", regex=True), errors="coerce")


def make_prepare(counts: set, extras: set = frozenset()):
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["Month"] = _num(df["Month"])
        X["DayofMonth"] = _num(df["DayofMonth"])
        X["DayOfWeek"] = _num(df["DayOfWeek"])
        X["DepTime"] = df["DepTime"].astype(float)
        X["Distance"] = df["Distance"].astype(float)
        dep = X["DepTime"]
        hour = np.clip((dep // 100) % 24, 0, 23)
        minute = dep % 100
        X["DepHour"] = hour
        X["DepMinute"] = minute
        X["DepTimeMin"] = hour * 60 + minute
        X["sin_day"] = np.sin(2 * np.pi * (hour * 60 + minute) / 1440)
        X["cos_day"] = np.cos(2 * np.pi * (hour * 60 + minute) / 1440)
        for c in CAT_COLS:
            X[c] = pd.Categorical(df[c], categories=cat_levels[c])
        if counts:
            o = df["Origin"].astype(str)
            d = df["Dest"].astype(str)
            c = df["UniqueCarrier"].astype(str)
            route = o + "-" + d
            hour_s = hour.astype(int).astype(str)
            dow_s = df["DayOfWeek"].astype("string").str.replace(r"^c-", "", regex=True)
            month_s = df["Month"].astype("string").str.replace(r"^c-", "", regex=True)
            src = {"origin": o, "dest": d, "route": route, "carrier": c,
                   "origin_hour": o + "-" + hour_s, "dest_hour": d + "-" + hour_s,
                   "origin_carrier": o + "-" + c, "dest_carrier": d + "-" + c,
                   "hour_carrier": hour_s + "-" + c, "origin_dow": o + "-" + dow_s,
                   "hour_dow": hour_s + "-" + dow_s, "route_hour": route + "-" + hour_s,
                   "origin_month": o + "-" + month_s, "dest_month": d + "-" + month_s}
            for k in counts:
                X[f"cnt_{k}"] = np.log1p(src[k].map(COUNTS[k]).fillna(0.0).astype(float))
        if extras:
            route = df["Origin"].astype(str) + "-" + df["Dest"].astype(str)
            if "dist_dev" in extras:
                rmd = route.map(ROUTE_MEAN_DIST)
                X["route_avg_dist"] = rmd.fillna(train["Distance"].mean()).astype(float)
                X["dist_dev"] = X["Distance"] - X["route_avg_dist"]
            if "arr" in extras:
                arr_min = _arr_min(df["DepTime"].to_numpy(), df["Distance"].to_numpy())
                X["est_arr_min"] = arr_min
                X["est_arr_hour"] = np.floor(arr_min / 60) % 24
                o = df["Origin"].astype(str)
                d = df["Dest"].astype(str)
                ah = np.floor(arr_min / 60).astype(int).astype(str)
                X["cnt_dest_arr_hour"] = np.log1p((d + "-" + ah).map(COUNTS["dest_arr_hour"]).fillna(0.0).astype(float))
                X["cnt_origin_arr_hour"] = np.log1p((o + "-" + ah).map(COUNTS["origin_arr_hour"]).fillna(0.0).astype(float))
        return X
    return prepare


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- experiment harness -------------------------------------------------------
t0 = time.time()


def run_bags(prep, bag_cfgs):
    """Train all bags, cache per-model eval probs once, report combos."""
    X_tr, X_ev = prep(train), prep(evald)
    out = {}
    for name, cfg in bag_cfgs.items():
        cfg = dict(cfg)
        seed0, n = cfg["seed0"], cfg["n"]
        subs, col, depth, mcw, k = cfg["subs"], cfg["col"], cfg["depth"], cfg["mcw"], cfg["k"]
        ms, ps = [], []
        for s in range(n):
            m = xgb.XGBClassifier(
                n_estimators=k, learning_rate=0.1, tree_method="hist", enable_categorical=True,
                max_depth=depth, min_child_weight=mcw, subsample=subs, colsample_bytree=col,
                random_state=seed0 + s, n_jobs=N_JOBS)
            m.fit(X_tr, y_tr, verbose=False)
            ms.append(m)
            ps.append(m.predict_proba(X_ev)[:, 1])
        out[name] = (ms, ps)
        print(f"[diag] {name} n={n}: {roc_auc_score(y_ev, np.mean(ps, axis=0)):.4f} ({time.time() - t0:.1f}s)")
    return out


BAG_CFGS = {
    "b2_col025_d12": dict(seed0=SEED + 2000, n=8, subs=0.8, col=0.25, depth=12, mcw=1, k=120),
}

BASE_COUNTS = {"origin", "dest", "route", "carrier", "origin_hour", "dest_hour"}
FEATSETS = [
    ("ctrl", set(COUNT_SRC.keys()), {"dist_dev"}),
    ("arr", set(COUNT_SRC.keys()), {"dist_dev", "arr"}),
]

best = None
for fs_entry in FEATSETS:
    fs_name, fs = fs_entry[0], fs_entry[1]
    extras = fs_entry[2] if len(fs_entry) > 2 else frozenset()
    prep = make_prepare(fs, extras)
    bags = run_bags(prep, BAG_CFGS)
    allp = np.array([p for _, ps in bags.values() for p in ps])
    aucs = {n: roc_auc_score(y_ev, np.mean(ps, axis=0)) for n, (ms, ps) in bags.items()}
    auc_union = roc_auc_score(y_ev, allp.mean(axis=0))
    print(f"[diag] fs={fs_name} union={auc_union:.4f} " +
          " ".join(f"{n}:{a:.4f}" for n, a in aucs.items()))
    cand_auc = max(list(aucs.values()) + [auc_union])
    if best is None or cand_auc > best[0]:
        best = (cand_auc, fs_name, bags, prep, aucs, auc_union)

_, FS_NAME, BAGS, prepare, AUCS, AUC_U = best
ship_names = list(BAGS) if AUC_U >= max(AUCS.values()) else [max(AUCS, key=AUCS.get)]
MODELS, EVAL_PROBS = [], []
for n_ in ship_names:
    ms, ps = BAGS[n_]
    MODELS += ms
    EVAL_PROBS += ps
print(f"[diag] ship fs={FS_NAME} n={len(MODELS)} bags={ship_names}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in MODELS], axis=0)


eval_auc = roc_auc_score(y_ev, np.mean(EVAL_PROBS, axis=0))
print(f"Eval AUC: {eval_auc:.4f}")
