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
from sklearn.model_selection import KFold, StratifiedKFold

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
cat_cols = [c for c in cat_cols if c not in ("Month", "DayofMonth")]
feature_cols = [c for c in feature_cols if c not in ("Month", "DayofMonth")]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# target-encoding keys (string keys built from raw columns)
TE_KEYS = ["Origin", "Dest", "UniqueCarrier", "route", "origin_hour", "dest_hour", "carrier_origin",
           "DayOfWeek", "arr_hour"]
TE_SMOOTH = 20.0
CNT_KEYS = ["Origin", "Dest", "UniqueCarrier", "route"]


def key_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "route":
        return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)
    if name == "origin_hour":
        hour = (df["DepTime"].to_numpy(dtype=np.int64) // 100).astype(str)
        return df["Origin"].astype(str) + "_" + hour
    if name == "arr_hour":
        tot = (df["DepTime"].to_numpy(dtype=np.int64) // 100) * 60 + df["DepTime"].to_numpy(dtype=np.int64) % 100
        arr = tot + df["Distance"].to_numpy(dtype=float) / 500.0 * 60.0
        return pd.Series((arr // 60).astype(int).astype(str), index=df.index)
    if name == "dest_hour":
        hour = (df["DepTime"].to_numpy(dtype=np.int64) // 100).astype(str)
        return df["Dest"].astype(str) + "_" + hour
    if name == "carrier_origin":
        return df["UniqueCarrier"].astype(str) + "_" + df["Origin"].astype(str)
    return df[name].astype(str)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    dep = df["DepTime"].to_numpy(dtype=float)
    hh = dep // 100.0
    mm = dep % 100.0
    tot = hh * 60.0 + mm
    X["dep_min"] = tot
    X["dep_hour"] = hh
    dur = df["Distance"].to_numpy(dtype=float) / 500.0 * 60.0
    arr = tot + dur
    X["arr_min"] = arr
    X["arr_hour"] = arr // 60.0
    X["dep_sin"] = np.sin(2.0 * np.pi * tot / 1440.0)
    X["dep_cos"] = np.cos(2.0 * np.pi * tot / 1440.0)
    dow = df["DayOfWeek"].astype(str)
    X["is_weekend"] = dow.isin(["c-6", "c-7"]).astype(int)
    for name in TE_KEYS:
        X["te_" + name] = key_series(df, name).map(te_maps[name]).fillna(te_prior).astype(float)
    for name in CNT_KEYS:
        X["cnt_" + name] = key_series(df, name).map(cnt_maps[name]).fillna(0.0).astype(float)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target encoding (out-of-fold for training, full-train map for inference) ---
ytr = to_y(train)
te_prior = float(ytr.mean())
te_maps = {}
oof_te = {}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name in TE_KEYS:
    keys = key_series(train, name)
    full = (pd.DataFrame({"k": keys, "y": ytr}).groupby("k")["y"]
            .agg(["sum", "count"]))
    te_maps[name] = (full["sum"] + te_prior * TE_SMOOTH) / (full["count"] + TE_SMOOTH)
    oof = np.full(len(train), te_prior)
    for tr_idx, va_idx in kf.split(keys):
        sub = pd.DataFrame({"k": keys.iloc[tr_idx], "y": ytr[tr_idx]}).groupby("k")["y"].agg(["sum", "count"])
        enc = (sub["sum"] + te_prior * TE_SMOOTH) / (sub["count"] + TE_SMOOTH)
        oof[va_idx] = keys.iloc[va_idx].map(enc).fillna(te_prior).to_numpy()
    oof_te[name] = oof
cnt_maps = {name: key_series(train, name).value_counts() for name in CNT_KEYS}

# --- model --------------------------------------------------------------------
CANDIDATES = [
    dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=5, reg_lambda=5),
    dict(n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.9, colsample_bytree=0.6,
         min_child_weight=20, reg_lambda=20),
    dict(n_estimators=200, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=20, reg_lambda=20),
    dict(n_estimators=400, max_depth=8, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7,
         min_child_weight=30, reg_lambda=30),
    dict(n_estimators=100, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
         min_child_weight=10, reg_lambda=10),
    dict(n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.9, colsample_bytree=0.5,
         min_child_weight=20, reg_lambda=20),
    dict(n_estimators=300, max_depth=8, learning_rate=0.03, subsample=0.9, colsample_bytree=0.7,
         min_child_weight=10, reg_lambda=10),
    dict(n_estimators=400, max_depth=10, learning_rate=0.02, subsample=0.8, colsample_bytree=0.6,
         min_child_weight=30, reg_lambda=50),
    dict(n_estimators=150, max_depth=8, learning_rate=0.05, subsample=0.9, colsample_bytree=0.6,
         min_child_weight=20, reg_lambda=20),
    dict(n_estimators=300, max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.6,
         min_child_weight=20, reg_lambda=20),
]


def make_model(cfg, seed=SEED):
    return xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=seed,
                             n_jobs=N_JOBS, **cfg)


Xtr = prepare(train)
for name in TE_KEYS:
    Xtr["te_" + name] = oof_te[name]  # out-of-fold replacing the full-train (leaky) encodings

t0 = time.time()
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
results = []
for cfg in CANDIDATES:
    scores = []
    for tr_idx, va_idx in skf.split(Xtr, ytr):
        m = make_model(cfg)
        m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        scores.append(roc_auc_score(ytr[va_idx], m.predict_proba(Xtr.iloc[va_idx])[:, 1]))
    sc = float(np.mean(scores))
    print(f"CV {sc:.4f} {cfg}")
    results.append((sc, cfg))
results.sort(key=lambda r: -r[0])
print(f"Best CV {results[0][0]:.4f} {results[0][1]}")

models = []
for sc, cfg in results[:2]:
    for s in range(SEED, SEED + 5):
        m = make_model(cfg, seed=s)
        m.fit(Xtr, ytr)
        models.append(m)
print(f"Ensemble of {len(models)} models")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = prepare(df)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
