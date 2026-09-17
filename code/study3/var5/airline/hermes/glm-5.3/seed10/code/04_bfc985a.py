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
cat_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

# --- target encoding machinery (fit on TRAIN ONLY) -----------------------------
TE_SMOOTH = 20.0
te_maps = {}  # key col -> (dict level -> smoothed mean)

# TE keys: raw columns and interactions, all computed from the raw frame
def te_keys(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "route":
        return (df["Origin"].astype(str) + "_" + df["Dest"].astype(str)).to_numpy()
    if name == "carrier_hour":
        return (df["UniqueCarrier"].astype(str) + "_" + (((df["DepTime"] // 100) % 24)).astype(str)).to_numpy()
    if name == "origin_hour":
        return (df["Origin"].astype(str) + "_" + (((df["DepTime"] // 100) % 24)).astype(str)).to_numpy()
    if name == "dest_hour":
        return (df["Dest"].astype(str) + "_" + (((df["DepTime"] // 100) % 24)).astype(str)).to_numpy()
    return df[name].to_numpy()


TE_COLS = ["UniqueCarrier", "Origin", "Dest", "route", "carrier_hour", "origin_hour", "dest_hour"]


def fit_fold(keys: np.ndarray, y: np.ndarray) -> dict:
    """Smoothed target means for one fold."""
    prior = float(y.mean())
    df = pd.DataFrame({"k": keys, "y": y})
    grp = df.groupby("k")["y"].agg(["sum", "count"])
    enc = (grp["sum"] + TE_SMOOTH * prior) / (grp["count"] + TE_SMOOTH)
    return enc.to_dict()


def apply_enc(keys: np.ndarray, mapping: dict) -> np.ndarray:
    return np.array([mapping.get(k, np.nan) for k in keys], dtype=float)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    t = X["DepTime"].to_numpy()
    hour = (t // 100) % 24
    minute = t % 100
    mins = hour * 60 + minute  # minutes since midnight
    X["hour"] = hour
    X["mins"] = mins
    day = 2 * np.pi * (mins / 1440.0)
    X["sin_day"] = np.sin(day)
    X["cos_day"] = np.cos(day)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=3000,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    enable_categorical=True,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
y_train = to_y(train)

# OOF target encoding: K-fold, encoders refit per fold so train rows get fold-local values
from sklearn.model_selection import KFold

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
te_oof = {c: np.full(len(train), np.nan) for c in TE_COLS}
key_cache = {c: te_keys(train, c) for c in TE_COLS}
for tr_idx, va_idx in kf.split(train):
    y_tr = y_train[tr_idx]
    for c in TE_COLS:
        mapping = fit_fold(key_cache[c][tr_idx], y_tr)
        te_oof[c][va_idx] = apply_enc(key_cache[c][va_idx], mapping)

# full-data encoders for eval/holdout
te_maps = {c: fit_fold(key_cache[c], y_train) for c in TE_COLS}

Xtr = pd.concat([prepare(train), pd.DataFrame({c + "_te": te_oof[c] for c in TE_COLS})], axis=1)
Xev = pd.concat([prepare(evald), pd.DataFrame({c + "_te": apply_enc(te_keys(evald, c), te_maps[c]) for c in TE_COLS})], axis=1)
model.fit(Xtr, y_train, eval_set=[(Xev, to_y(evald))], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    block = pd.DataFrame({c + "_te": apply_enc(te_keys(df, c), te_maps[c]) for c in TE_COLS}, index=df.index)
    return model.predict_proba(pd.concat([prepare(df), block], axis=1))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
