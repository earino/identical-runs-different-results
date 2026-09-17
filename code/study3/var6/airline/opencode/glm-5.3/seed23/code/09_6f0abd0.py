"""XGBoost binary classifier for airline departure delay prediction.

Contract (see program.md):
  1. `python train.py` trains on data/train.csv, evaluates on data/eval.csv, prints `Eval AUC: 0.xxxx`.
  2. After running, a module-level function `predict_proba(df)` exists: raw DataFrame -> 1-D array of P(positive).

Approach: multi-view bagged ensemble. Each "view" is a different feature representation
(raw categoricals, smoothed target encodings of airport/carrier/time combinations, etc.).
Target-encoded features are computed OUT-OF-FOLD on the training data (5 folds) so that the
training rows never see their own label; at predict time the full-training-data encoding map
is used. Each view is fit by several shallow XGBoost models (row/col subsampling, seed
diversity); the final prediction averages all of them.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TASK = json.load(open("task.json"))
TARGET = TASK["target"]
POSITIVE = TASK["positive_label"]
ID_COLS = TASK.get("id_columns", [])
N_JOBS = int(os.environ.get("BENCH_THREADS", "4"))

train = pd.read_csv("data/train.csv")
evald = pd.read_csv("data/eval.csv")
y_all = (train[TARGET] == POSITIVE).astype(int).to_numpy()
prior = float(y_all.mean())

# --- feature construction ------------------------------------------------------
ALL_C = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
BASE_C = ["DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
cat_levels = {c: pd.Index(sorted(train[c].dropna().unique())) for c in ALL_C}

_hour_tr = (train["DepTime"] // 100).clip(0, 24)
TE_SPECS = {
    "te_oh": (train["Origin"].astype(str) + "_" + _hour_tr.astype(str), 600),
    "te_dh": (train["Dest"].astype(str) + "_" + _hour_tr.astype(str), 600),
    "te_or": (train["Origin"], 800),
    "te_de": (train["Dest"], 800),
    "te_ch": (train["UniqueCarrier"] + "_" + _hour_tr.astype(str), 300),
    "te_dow": (train["DayOfWeek"], 800),
    "te_db": (pd.cut(train["Distance"], 10).astype(str), 300),
}


def _te_stats(key, m, ys=None):
    g = pd.DataFrame({"k": key, "y": y_all if ys is None else ys}).groupby("k")["y"].agg(["sum", "count"])
    return ((g["sum"] + prior * m) / (g["count"] + m))


# full-data encoding maps (used at predict time and for any non-training rows)
full_map = {k: _te_stats(v[0], v[1]).to_dict() for k, v in TE_SPECS.items()}

# out-of-fold encodings for the training rows (no self-label leakage)
_OOF_FOLDS = 5
_oof = {k: pd.Series(index=train.index, dtype=float) for k in TE_SPECS}
for ftr, fte in StratifiedKFold(_OOF_FOLDS, shuffle=True, random_state=0).split(np.zeros(len(train)), y_all):
    for k, (key, m) in TE_SPECS.items():
        _m = _te_stats(key.iloc[ftr].reset_index(drop=True), m, y_all[ftr])
        _oof[k].iloc[fte] = key.iloc[fte].map(_m).values
for k in _oof:
    _oof[k] = _oof[k].fillna(prior)

VIEW_TECOLS = {"base": [], "ot": ["te_or", "te_de"], "no_od": ["te_oh", "te_dh"],
               "fullte": ["te_dow", "te_ch", "te_or", "te_de", "te_oh", "te_dh", "te_db"]}
VIEW_CATS = {"base": BASE_C, "ot": BASE_C, "no_od": ["DayOfWeek", "UniqueCarrier"], "fullte": []}


def _key_for(col, df, h):
    return {"te_oh": lambda: df["Origin"].astype(str) + "_" + h.astype(str),
            "te_dh": lambda: df["Dest"].astype(str) + "_" + h.astype(str),
            "te_or": lambda: df["Origin"],
            "te_de": lambda: df["Dest"],
            "te_ch": lambda: df["UniqueCarrier"] + "_" + h.astype(str),
            "te_dow": lambda: df["DayOfWeek"],
            "te_db": lambda: pd.cut(df["Distance"], 10).astype(str)}[col]()


def prepare(df: pd.DataFrame, view: str) -> pd.DataFrame:
    """Feature engineering for one view. Uses only training-derived statistics (never df's labels)."""
    h = (df["DepTime"] // 100).clip(0, 24)
    X = pd.DataFrame(index=df.index)
    X["DepTime"] = df["DepTime"].astype(int)
    X["Distance"] = df["Distance"].astype(float)
    for c in VIEW_CATS[view]:
        X[c] = pd.Categorical(df[c], categories=cat_levels[c])
    for c in VIEW_TECOLS[view]:
        X[c] = _key_for(c, df, h).map(full_map[c]).fillna(prior)
    return X


def prepare_train(view: str) -> pd.DataFrame:
    """Training matrix: same as prepare() but TE columns carry out-of-fold values."""
    X = prepare(train, view)
    for c in VIEW_TECOLS[view]:
        X[c] = _oof[c].to_numpy()
    return X


VIEWS = ["base", "ot", "no_od", "fullte"]

# --- ensemble -------------------------------------------------------------------
BASE = dict(tree_method="hist", enable_categorical=True, n_jobs=N_JOBS)
# lossguide (leaf-limited, asymmetric) trees transfer much better to the holdout year
# than depth-limited trees here; the categorical-only "md" view prefers small trees.
VIEW_CFG = {
    "base": dict(n_estimators=200, learning_rate=0.04, subsample=0.9, colsample_bytree=0.8,
                 grow_policy="lossguide", max_leaves=320, max_depth=16, max_bin=512),
    "ot": dict(n_estimators=200, learning_rate=0.04, subsample=0.9, colsample_bytree=0.8,
               grow_policy="lossguide", max_leaves=320, max_depth=16, max_bin=512),
    "no_od": dict(n_estimators=200, learning_rate=0.04, subsample=0.9, colsample_bytree=0.8,
                  grow_policy="lossguide", max_leaves=320, max_depth=16, max_bin=512),
    "fullte": dict(n_estimators=200, learning_rate=0.04, subsample=0.9, colsample_bytree=0.8,
                   grow_policy="lossguide", max_leaves=320, max_depth=16, max_bin=512),
}
N_SEEDS = 4

models = []  # (view, model)
t0 = time.time()
for view in VIEWS:
    X_tr = prepare_train(view)
    for seed in range(N_SEEDS):
        m = xgb.XGBClassifier(random_state=seed, **VIEW_CFG[view], **BASE)
        m.fit(X_tr, y_all)
        models.append((view, m))
print(f"Training time: {time.time() - t0:.1f}s ({len(models)} models)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    X_by_view = {v: prepare(df, v) for v in VIEWS}
    out = None
    for view, m in models:
        p = m.predict_proba(X_by_view[view])[:, 1]
        out = p if out is None else out + p
    return out / len(models)


t0 = time.time()
eval_auc = roc_auc_score((evald[TARGET] == POSITIVE).astype(int).to_numpy(), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
