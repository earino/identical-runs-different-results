"""XGBoost binary classifier on airline delays. THIS IS THE ONLY FILE THE AGENT EDITS.

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

y = (train[TARGET] == POSITIVE).astype(int).to_numpy()
ye = (evald[TARGET] == POSITIVE).astype(int).to_numpy()

# trimmed feature set (Month/DayofMonth removed: 2005-specific noise)
kept_cols = [c for c in feature_cols if c not in ("Month", "DayofMonth")]
kept_lv = {c: cat_levels[c] for c in cat_cols if c in kept_cols}

# target encodings fit on train only
p_bar = float(y.mean())


def _te(keys: pd.Series, m: float = 20.0) -> dict:
    g = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + m * p_bar) / (g["count"] + m)).to_dict()


_hh_tr = (train["DepTime"].fillna(-1).astype(int) // 100).astype(str)
TE = {
    "car_hour": _te(train["UniqueCarrier"].astype(str) + "_" + _hh_tr),
    "dow_hour": _te(train["DayOfWeek"].astype(str) + "_" + _hh_tr),
    "carrier": _te(train["UniqueCarrier"]),
    "dow": _te(train["DayOfWeek"]),
    "origin": _te(train["Origin"]),
    "dest": _te(train["Dest"]),
}

base = dict(
    learning_rate=0.1,
    max_depth=6,
    n_estimators=120,
    tree_method="hist",
    enable_categorical=True,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=N_JOBS,
)
K = 8


def prep_extra(df: pd.DataFrame, extra: str) -> pd.DataFrame:
    X = df[kept_cols].copy()
    for c in kept_lv:
        X[c] = pd.Categorical(X[c], categories=kept_lv[c])
    tod = (df["DepTime"].fillna(-1).astype(int) // 100 * 60 + df["DepTime"].fillna(-1).astype(int) % 100).astype(float)
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    if extra == "route":
        rt_tr = (train["Origin"].astype(str) + "_" + train["Dest"].astype(str))
        route_levels = pd.Index(sorted(rt_tr.unique()))
        X["route"] = pd.Categorical(df["Origin"].astype(str) + "_" + df["Dest"].astype(str), categories=route_levels)
    if extra == "-dist":
        X = X.drop(columns=["Distance"])
    if extra == "-origin":
        X = X.drop(columns=["Origin"])
    if extra == "-dest":
        X = X.drop(columns=["Dest"])
    return X


def run_variant(extra, label, cfg=None, K_=8):
    X, Xe = prep_extra(train, extra), prep_extra(evald, extra)
    c = cfg or base
    preds = [xgb.XGBClassifier(random_state=SEED + k, **c).fit(X, y).predict_proba(Xe)[:, 1] for k in range(K_)]
    auc = roc_auc_score(ye, np.mean(preds, axis=0))
    print(f"{label}: {auc:.4f}")
    return auc, extra, c, K_


t0 = time.time()
results = [
    run_variant("none", "d16 n120 K8", cfg={**base, "max_depth": 16}),
]
best_auc, best_extra, best_cfg, best_K = max(results, key=lambda r: r[0])
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {best_auc:.4f}")

models = [
    xgb.XGBClassifier(random_state=SEED + k, **best_cfg).fit(prep_extra(train, best_extra), y)
    for k in range(best_K)
]


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    Xf = prep_extra(df, best_extra)
    return np.mean([m.predict_proba(Xf)[:, 1] for m in models], axis=0)
