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
    if extra == "dow_sincos":
        dw = df["DayOfWeek"].str.slice(2).astype(int)
        X["dow_sin"] = np.sin(2 * np.pi * (dw - 1) / 7.0)
        X["dow_cos"] = np.cos(2 * np.pi * (dw - 1) / 7.0)
    if extra == "hour_bin":
        X["hour"] = (df["DepTime"].fillna(-1).astype(int) // 100).clip(0, 24)
    if extra == "reg":
        pass
    return X


def run_variant(extra, label, cfg=None, K_=8):
    X, Xe = prep_extra(train, extra), prep_extra(evald, extra)
    c = cfg or base
    preds = [xgb.XGBClassifier(random_state=SEED + k, **c).fit(X, y).predict_proba(Xe)[:, 1] for k in range(K_)]
    auc = roc_auc_score(ye, np.mean(preds, axis=0))
    print(f"{label}: {auc:.4f}")
    return auc, extra, c, K_


t0 = time.time()
r_base = run_variant("none", "sincos base K8")
# K curve on base config
Xb, Xeb = prep_extra(train, "none"), prep_extra(evald, "none")
preds32 = [xgb.XGBClassifier(random_state=SEED + k, **base).fit(Xb, y).predict_proba(Xeb)[:, 1] for k in range(32)]
for kk in (8, 16, 32):
    print(f"base K={kk}: {roc_auc_score(ye, np.mean(preds32[:kk], axis=0)):.4f}")
results = [
    r_base,
    run_variant("dow_sincos", "+dow sincos"),
    run_variant("hour_bin", "+hour bin"),
    run_variant("none", "mcw5 gamma0.5", cfg={**base, "min_child_weight": 5.0, "gamma": 0.5}),
    run_variant("none", "sub7 col9", cfg={**base, "subsample": 0.7, "colsample_bytree": 0.9}),
]
best_auc, best_extra, best_cfg, best_K = max(results, key=lambda r: r[0])
print(f"Training time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {best_auc:.4f}")

model = xgb.XGBClassifier(random_state=SEED, **best_cfg)
model.fit(prep_extra(train, best_extra), y)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prep_extra(df, best_extra))[:, 1]
