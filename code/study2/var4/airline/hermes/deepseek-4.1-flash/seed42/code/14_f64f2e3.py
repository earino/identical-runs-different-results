"""Baseline XGBoost binary classifier. THIS IS THE ONLY FILE THE AGENT EDITS.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).
  3. All feature engineering lives in prepare(); every statistic it uses is fit on data/train.csv only.
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

# --- features -----------------------------------------------------------------
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]
obj_cols = [c for c in feature_cols if pd.api.types.is_object_dtype(train[c]) or pd.api.types.is_string_dtype(train[c])]
cat_cols = [c for c in obj_cols if train[c].nunique() <= 1000]
feature_cols = [c for c in feature_cols if c not in obj_cols or c in cat_cols]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in cat_cols}

Y_TRAIN = (train[TARGET] == POSITIVE).astype(int).to_numpy()
PRIOR = float(Y_TRAIN.mean())


def add_keys(df: pd.DataFrame) -> dict:
    """Raw -> string keys for target encoding (no train statistics involved)."""
    dep = df["DepTime"].astype(float).to_numpy()
    dep = np.where(dep >= 2400, dep - 2400.0, dep)
    hour = np.floor(dep / 100.0).astype(int).astype(str)
    origin = df["Origin"].astype(str)
    return {
        "Origin": origin,
        "Dest": df["Dest"].astype(str),
        "UniqueCarrier": df["UniqueCarrier"].astype(str),
        "Origin_hour": origin + "|" + hour,
    }


# --- target encoding: smoothed means, fit on TRAIN ONLY ------------------------
TE_SPEC = {"Origin": ("Origin", 50.0), "Dest": ("Dest", 50.0),
           "UniqueCarrier": ("UniqueCarrier", 50.0), "Origin_hour": ("Origin_hour", 30.0)}
ACTIVE_TE = ["Origin", "Dest", "UniqueCarrier", "Origin_hour"]
_KEYS_TRAIN = add_keys(train)


def _te_stats(keys, y, k):
    s = pd.Series(y, index=keys.index).groupby(keys.to_numpy()).agg(["sum", "count"])
    return ((s["sum"] + PRIOR * k) / (s["count"] + k)).to_dict()


te_maps, oof_te = {}, {}
for _n, (_key, _k) in TE_SPEC.items():
    _ks = _KEYS_TRAIN[_key]
    te_maps[_n] = _te_stats(_ks, Y_TRAIN, _k)
    _oof = np.full(len(train), PRIOR)
    for _tr, _va in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
        _m = _te_stats(_ks.iloc[_tr], Y_TRAIN[_tr], _k)
        _oof[_va] = _ks.iloc[_va].map(_m).fillna(PRIOR).to_numpy()
    oof_te[_n] = _oof


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    keys = add_keys(df)
    for n in ACTIVE_TE:
        X[f"te_{n}"] = keys[TE_SPEC[n][0]].map(te_maps[n]).fillna(PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr = prepare(train)
for n in ACTIVE_TE:  # train rows use out-of-fold encodings
    Xtr[f"te_{n}"] = oof_te[n]
ytr = to_y(train)
Xev, yev = prepare(evald), to_y(evald)

BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)

# --- DIAGNOSTIC: refine around alpha=10 ---------------------------------------
for _p in [dict(max_depth=4, learning_rate=0.02, n_estimators=600, reg_alpha=10.0),
           dict(max_depth=4, learning_rate=0.02, n_estimators=600, reg_alpha=8.0),
           dict(max_depth=4, learning_rate=0.02, n_estimators=600, reg_alpha=12.0),
           dict(max_depth=4, learning_rate=0.02, n_estimators=600, reg_alpha=15.0),
           dict(max_depth=4, learning_rate=0.02, n_estimators=900, reg_alpha=10.0),
           dict(max_depth=4, learning_rate=0.02, n_estimators=1200, reg_alpha=10.0),
           dict(max_depth=5, learning_rate=0.02, n_estimators=600, reg_alpha=10.0),
           dict(max_depth=5, learning_rate=0.02, n_estimators=900, reg_alpha=10.0),
           dict(max_depth=4, learning_rate=0.01, n_estimators=1200, reg_alpha=10.0),
           dict(max_depth=4, learning_rate=0.02, n_estimators=600, reg_alpha=10.0, reg_lambda=5.0)]:
    _m = xgb.XGBClassifier(random_state=SEED, **{**BASE, **_p})
    _m.fit(Xtr, ytr)
    print(f"diag a10 {str(_p):96s} eval_auc={roc_auc_score(yev, _m.predict_proba(Xev)[:, 1]):.4f}")

# --- model: L1-regularized reference ------------------------------------------
model = xgb.XGBClassifier(max_depth=4, learning_rate=0.02, n_estimators=600, reg_alpha=10.0,
                          random_state=SEED, **BASE)
t0 = time.time()
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
