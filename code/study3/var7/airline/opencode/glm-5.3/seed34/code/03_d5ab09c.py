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

# smoothed target encodings fitted on TRAIN only; train rows get out-of-fold values
PRIOR = float(y_train.mean())
TE_SPECS = [("carrier_te", "UniqueCarrier", None, 20), ("origin_te", "Origin", None, 100),
            ("dest_te", "Dest", None, 100), ("route_te", None, "route", 500)]


def _route(df):
    return df["Origin"] + "_" + df["Dest"]


def _smooth_te(keys, y, m):
    g = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return (g["sum"] + m * PRIOR) / (g["count"] + m)


te_maps = {}
te_oof = {}
for name, col, rcol, m in TE_SPECS:
    ktr = (_route(train) if rcol else train[col]).to_numpy()
    te_maps[name] = _smooth_te(ktr, y_train, m)
    oof = np.full(len(ktr), PRIOR)
    for itr, iho in KFold(5, shuffle=True, random_state=1).split(ktr):
        v = _smooth_te(ktr[itr], y_train[itr], m)
        oof[iho] = pd.Series(ktr[iho]).map(v).fillna(PRIOR).to_numpy()
    te_oof[name] = oof


def prepare(df: pd.DataFrame, with_te: bool = False) -> pd.DataFrame:
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
    for k in (1, 2, 3):
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
    if with_te:
        if len(df) == len(train) and df.index.equals(train.index) and df[CAT_COLS[0]].equals(train[CAT_COLS[0]]):
            for name, col, rcol, m in TE_SPECS:
                X[name] = te_oof[name]
        else:
            for name, col, rcol, m in TE_SPECS:
                keys = (_route(df) if rcol else df[col]).to_numpy()
                X[name] = pd.Series(keys).map(te_maps[name]).fillna(PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- models: small diverse XGBoost ensemble ------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
CONFIGS = [
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), False),
    (dict(n_estimators=600, max_depth=9, learning_rate=0.05, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), False),
    (dict(n_estimators=300, max_depth=8, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), False),
    (dict(n_estimators=300, max_depth=14, learning_rate=0.1, reg_alpha=5.0, min_child_weight=10, colsample_bytree=0.7), False),
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7, gamma=1.0), False),
    (dict(n_estimators=300, max_depth=9, learning_rate=0.1, reg_alpha=10.0, min_child_weight=10, colsample_bytree=0.7), True),
]

t0 = time.time()
X_plain = prepare(train)
X_te = prepare(train, with_te=True)
models = []
for params, use_te in CONFIGS:
    m = xgb.XGBClassifier(**params, **BASE)
    m.fit(X_te if use_te else X_plain, y_train)
    models.append((m, use_te))
print(f"Training time: {time.time() - t0:.1f}s for {len(models)} models")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xs = {False: prepare(df), True: prepare(df, with_te=True)}
    ps = [m.predict_proba(Xs[ute])[:, 1] for m, ute in models]
    return np.mean(ps, axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
