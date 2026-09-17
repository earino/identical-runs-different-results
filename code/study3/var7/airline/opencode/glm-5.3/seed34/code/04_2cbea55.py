"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Model: equal-weight ensemble of 10 XGBoost models. Members differ in depth / rounds / gamma /
tod-harmonics count, and in whether smoothed 5-fold out-of-fold target encodings of
carrier/origin/dest/route (fitted on train only) are appended as numeric features.
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
y_train = (train[TARGET] == POSITIVE).astype(int).to_numpy()

# --- features -----------------------------------------------------------------
CAT_COLS = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in CAT_COLS}
hod_levels = pd.Index(sorted((train["DepTime"].astype(int) // 100).unique())).astype(str)

# smoothed target encodings fitted on TRAIN only; training rows get out-of-fold values,
# rows seen at predict time get the full-train maps (unseen keys fall back to the prior).
PRIOR = float(y_train.mean())
TE_SPECS = [("carrier_te", "UniqueCarrier", 20), ("origin_te", "Origin", 100),
            ("dest_te", "Dest", 100), ("route_te", None, 500)]


def _route(df):
    return df["Origin"] + "_" + df["Dest"]


def _smooth_te(keys, y, m):
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * PRIOR) / (g["count"] + m)


te_maps, te_oof = {}, {}
for name, col, m in TE_SPECS:
    ktr = (_route(train) if col is None else train[col]).to_numpy()
    te_maps[name] = _smooth_te(ktr, y_train, m)
    oof = np.full(len(ktr), PRIOR)
    for itr, iho in KFold(5, shuffle=True, random_state=1).split(ktr):
        v = _smooth_te(ktr[itr], y_train[itr], m)
        oof[iho] = pd.Series(ktr[iho]).map(v).fillna(PRIOR).to_numpy()
    te_oof[name] = oof


def _base(df, harm):
    X = pd.DataFrame(index=df.index)
    for c in CAT_COLS:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    month = df["Month"].str.slice(2).astype(int)
    day = df["DayofMonth"].str.slice(2).astype(int)
    dep = df["DepTime"].astype(int)
    hh, mm = dep // 100, dep % 100
    tod = hh * 60 + mm
    X["dep"] = dep
    X["hh"] = hh
    X["tod"] = tod
    for k in range(1, harm + 1):
        X[f"tod_sin{k}"] = np.sin(2 * k * np.pi * tod / 1440)
        X[f"tod_cos{k}"] = np.cos(2 * k * np.pi * tod / 1440)
    X["hod_cat"] = pd.Categorical(hh.astype(str), categories=hod_levels)
    X["dist"] = df["Distance"]
    X["dist_hh"] = (df["Distance"].to_numpy() / 1000.0) * hh.to_numpy()
    doy = month * 31 + day
    X["doy"] = doy
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["mm"] = mm
    return X


def _add_te(X, df, te_vals):
    for name, col, m in TE_SPECS:
        X[name] = te_vals[name]
    return X


def prepare(df, harm=3, with_te=False, oof=None):
    X = _base(df, harm)
    if with_te:
        if oof is not None:
            return _add_te(X, df, oof)
        vals = {}
        for name, col, m in TE_SPECS:
            keys = (_route(df) if col is None else df[col]).to_numpy()
            vals[name] = pd.Series(keys).map(te_maps[name]).fillna(PRIOR).to_numpy()
        return _add_te(X, df, vals)
    return X


def to_y(df):
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- ensemble members ----------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
CONFIGS = [
    # (params, harm, with_te)
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), 3, False),
    (dict(n_estimators=600, max_depth=9, learning_rate=0.05, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), 3, False),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), 3, False),
    (dict(n_estimators=300, max_depth=14, learning_rate=0.1, reg_alpha=5.0, min_child_weight=10, colsample_bytree=0.7), 3, False),
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7, gamma=1.0), 3, False),
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), 3, True),
    (dict(n_estimators=600, max_depth=9, learning_rate=0.05, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), 3, True),
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7, gamma=1.0), 3, True),
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), 6, True),
    (dict(n_estimators=600, max_depth=9, learning_rate=0.05, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), 6, True),
]

t0 = time.time()
X_cache = {}
X_cache[(3, False)] = prepare(train, 3, False)
X_cache[(3, True)] = prepare(train, 3, True, oof=te_oof)
X_cache[(6, True)] = prepare(train, 6, True, oof=te_oof)
models = []
for params, harm, use_te in CONFIGS:
    m = xgb.XGBClassifier(**params, **BASE)
    m.fit(X_cache[(harm, use_te)], y_train)
    models.append((m, harm, use_te))
print(f"Training time: {time.time() - t0:.1f}s for {len(models)} models")


def predict_proba(df):
    out = []
    for m, harm, use_te in models:
        out.append(m.predict_proba(prepare(df, harm, use_te))[:, 1])
    return np.mean(out, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
