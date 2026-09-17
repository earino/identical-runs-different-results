"""XGBoost binary classifier for airline departure delay. THIS IS THE ONLY FILE THE AGENT EDITS.

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

RAW = [c for c in train.columns if c not in ID_COLS + [TARGET]]

# --- categorical vocabularies, fitted on TRAIN ONLY ---------------------------
CAT_COLS = ["UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].astype(str).unique())) for c in CAT_COLS}

CN_COLS = ["Month", "DayofMonth", "DayOfWeek"]

TE_KEYS = ["carrier", "origin", "dest", "route"]
FREQ_KEYS = ["carrier", "origin", "dest", "route", "origin_hour", "carrier_hour", "dest_hour"]
SMOOTH = 50.0


def _cn(s: pd.Series) -> pd.Series:
    """'c-7' -> 7.0 (missing/unparseable -> NaN)."""
    return pd.to_numeric(s.astype(str).str.slice(2), errors="coerce")


def _str(s: pd.Series) -> pd.Series:
    return s.astype(str)


def _hour(df: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    return (t // 100).clip(lower=0, upper=26).fillna(0).astype(int).astype(str)


def _key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All grouping keys, derived from raw columns only."""
    k = pd.DataFrame(index=df.index)
    k["carrier"] = _str(df["UniqueCarrier"])
    k["origin"] = _str(df["Origin"])
    k["dest"] = _str(df["Dest"])
    k["route"] = k["origin"] + "_" + k["dest"]
    h = _hour(df)
    k["origin_hour"] = k["origin"] + "_" + h
    k["dest_hour"] = k["dest"] + "_" + h
    k["carrier_hour"] = k["carrier"] + "_" + h
    return k


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- target / count statistics, fitted on TRAIN ONLY --------------------------
train_y = to_y(train)
PRIOR = float(train_y.mean())
KEY_TRAIN = _key_frame(train)


def _fit_te_maps(keys: pd.DataFrame, y: np.ndarray, cols):
    """Smoothed P(delay | key) from the given labelled rows."""
    prior = float(y.mean())
    maps = {}
    for name in cols:
        g = pd.DataFrame({"k": keys[name].to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
        maps[name] = ((g["sum"] + prior * SMOOTH) / (g["count"] + SMOOTH)).to_dict()
    return maps, prior


TE_MAPS, _ = _fit_te_maps(KEY_TRAIN, train_y, TE_KEYS)

# frequency (log1p count) of each key in the training year
FREQ_MAPS = {name: KEY_TRAIN[name].value_counts().to_dict() for name in FREQ_KEYS}


def _apply_te(k: pd.DataFrame, maps, prior: float) -> pd.DataFrame:
    out = pd.DataFrame(index=k.index)
    for name in maps:
        out["te_" + name] = k[name].map(maps[name]).fillna(prior).astype(float)
    return out


def _apply_freq(k: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=k.index)
    for name in FREQ_KEYS:
        out["f_" + name] = np.log1p(k[name].map(FREQ_MAPS[name]).fillna(0.0).astype(float))
    return out


def _oof_te(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold target encodings for the TRAINING frame (avoids self-leakage)."""
    y = to_y(df)
    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, n_splits, size=len(df))
    keys = _key_frame(df)
    out = pd.DataFrame(index=df.index, columns=["te_" + c for c in TE_KEYS], dtype=float)
    for f in range(n_splits):
        tr, va = fold != f, fold == f
        m, prior = _fit_te_maps(keys.loc[tr], y[tr], TE_KEYS)
        out.loc[va, :] = _apply_te(keys.loc[va], m, prior).to_numpy()
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = pd.DataFrame(index=df.index)
    for c in CN_COLS:
        X[c] = _cn(df[c])
    t = pd.to_numeric(df["DepTime"], errors="coerce")
    tod = (t // 100) * 60 + (t % 100)  # minutes since midnight (can exceed 1440)
    X["tod"] = tod
    X["tod_sin"] = np.sin(2 * np.pi * tod / 1440.0)
    X["tod_cos"] = np.cos(2 * np.pi * tod / 1440.0)
    X["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    for c in CAT_COLS:
        X[c] = pd.Categorical(_str(df[c]), categories=cat_levels[c])  # unseen levels -> NaN
    k = _key_frame(df)
    X = pd.concat([X, _apply_te(k, TE_MAPS, PRIOR), _apply_freq(k)], axis=1)
    return X


# --- model --------------------------------------------------------------------
X = prepare(train)
y = to_y(train)
# training path uses out-of-fold target encodings so the model cannot read its own labels
_oof = _oof_te(train)
X[_oof.columns] = _oof

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X))
n_val = int(0.1 * len(X))
val_idx, fit_idx = perm[:n_val], perm[n_val:]
Xfit, yfit = X.iloc[fit_idx], y[fit_idx]
Xval, yval = X.iloc[val_idx], y[val_idx]

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=5.0,
    tree_method="hist",
    enable_categorical=True,
    max_cat_to_onehot=8,
    n_jobs=N_JOBS,
    random_state=SEED,
    eval_metric="auc",
    early_stopping_rounds=30,
)

t0 = time.time()
model.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
print(f"Training time: {time.time() - t0:.1f}s  best_iter={model.best_iteration}")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


print(f"Val AUC (2005 holdout): {roc_auc_score(yval, model.predict_proba(Xval)[:, 1]):.4f}")
t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
