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

# --- target encoding: smoothed means, fit on TRAIN ONLY ------------------------
TE_COLS = ["Origin", "Dest", "UniqueCarrier"]
TE_SMOOTH = 50.0
_te_keys = {c: train[c].astype(str) for c in TE_COLS}


def _te_stats(keys: pd.Series, y: np.ndarray, k: float) -> dict:
    s = pd.Series(y, index=keys.index).groupby(keys.to_numpy()).agg(["sum", "count"])
    return ((s["sum"] + PRIOR * k) / (s["count"] + k)).to_dict()


te_maps = {c: _te_stats(_te_keys[c], Y_TRAIN, TE_SMOOTH) for c in TE_COLS}

# out-of-fold encoded values for the training rows themselves (avoids self-leakage)
oof_te = {c: np.full(len(train), PRIOR) for c in TE_COLS}
for _tr_idx, _va_idx in KFold(n_splits=5, shuffle=True, random_state=SEED).split(train):
    for c in TE_COLS:
        m = _te_stats(_te_keys[c].iloc[_tr_idx], Y_TRAIN[_tr_idx], TE_SMOOTH)
        oof_te[c][_va_idx] = _te_keys[c].iloc[_va_idx].map(m).fillna(PRIOR).to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    for c in TE_COLS:
        X[f"te_{c}"] = df[c].astype(str).map(te_maps[c]).fillna(PRIOR).to_numpy()
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


Xtr, ytr = prepare(train), to_y(train)
for c in TE_COLS:  # train rows use out-of-fold encodings
    Xtr[f"te_{c}"] = oof_te[c]
Xev, yev = prepare(evald), to_y(evald)

# --- DIAGNOSTIC: does TE help, and which columns carry it? --------------------
ALL_TE = ["te_" + c for c in TE_COLS]
for _cols in (["Origin"], ["Dest"], ["UniqueCarrier"], ["Origin", "Dest"], TE_COLS):
    _drop = ["te_" + c for c in TE_COLS if c not in _cols]
    _A, _B = Xtr.drop(columns=_drop), Xev.drop(columns=_drop)
    _m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, tree_method="hist",
                           enable_categorical=True, random_state=SEED, n_jobs=N_JOBS)
    _m.fit(_A, ytr)
    print(f"diag TE={_cols} eval_auc={roc_auc_score(yev, _m.predict_proba(_B)[:, 1]):.4f}")

# --- model --------------------------------------------------------------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
model.fit(Xtr, ytr)
print(f"Training time: {time.time() - t0:.1f}s")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(yev, predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
