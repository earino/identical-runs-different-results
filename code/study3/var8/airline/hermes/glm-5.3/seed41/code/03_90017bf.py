"""XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

This experiment: probe (a) seed ensembles of the best config, (b) hour/cyclic features with the best config.
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

BEST = dict(n_estimators=300, max_depth=3, learning_rate=0.05, reg_lambda=5.0)


def prepare(df: pd.DataFrame, hour: bool = False) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows.
    X = df[feature_cols].copy()
    if hour:
        dep = pd.to_numeric(X["DepTime"], errors="coerce").astype("float64")
        h = np.floor(dep / 100.0)
        X["DepHour"] = h
        X["HourSin"] = np.sin(2 * np.pi * h / 24.0)
        X["HourCos"] = np.cos(2 * np.pi * h / 24.0)
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- probe ---------------------------------------------------------------------
ytr, yev = to_y(train), to_y(evald)

t0 = time.time()
variants = []  # (auc, name, predict_fn)


def fit_one(Xtr, seed):
    m = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, random_state=seed,
                          n_jobs=N_JOBS, **BEST)
    m.fit(Xtr, ytr)
    return m


# a) single seed control
Xtr0, Xev0 = prepare(train), prepare(evald)
m = fit_one(Xtr0, SEED)
variants.append((roc_auc_score(yev, m.predict_proba(Xev0)[:, 1]), "single", lambda df: m.predict_proba(prepare(df))[:, 1]))
print(f"[probe] single: {variants[-1][0]:.4f}  ({time.time() - t0:.1f}s)")

# b) seed ensemble (9 seeds)
members = []
for s in range(9):
    members.append(fit_one(Xtr0, 100 + s))
variants.append((roc_auc_score(yev, np.mean([mm.predict_proba(Xev0)[:, 1] for mm in members], axis=0)),
                 "seedens9", lambda df: np.mean([mm.predict_proba(prepare(df))[:, 1] for mm in members], axis=0)))
print(f"[probe] seedens9: {variants[-1][0]:.4f}  ({time.time() - t0:.1f}s)")

# c) hour features single
XtrH, XevH = prepare(train, hour=True), prepare(evald, hour=True)
mh = fit_one(XtrH, SEED)
variants.append((roc_auc_score(yev, mh.predict_proba(XevH)[:, 1]), "hour_single",
                 lambda df: mh.predict_proba(prepare(df, hour=True))[:, 1]))
print(f"[probe] hour_single: {variants[-1][0]:.4f}  ({time.time() - t0:.1f}s)")

# d) hour features + seed ensemble
membersH = []
for s in range(9):
    membersH.append(fit_one(XtrH, 100 + s))
variants.append((roc_auc_score(yev, np.mean([mm.predict_proba(XevH)[:, 1] for mm in membersH], axis=0)),
                 "hour_seedens9",
                 lambda df: np.mean([mm.predict_proba(prepare(df, hour=True))[:, 1] for mm in membersH], axis=0)))
print(f"[probe] hour_seedens9: {variants[-1][0]:.4f}  ({time.time() - t0:.1f}s)")

best_auc, best_name, best_fn = max(variants, key=lambda r: r[0])
print(f"[probe] selected {best_name}")
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return best_fn(df)


eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval AUC: {eval_auc:.4f}")
