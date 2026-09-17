"""XGBoost binary classifier for the airline delay task. THE ONLY FILE THE AGENT EDITS.

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
from sklearn.model_selection import train_test_split

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

# smoothed target encoding, fit on training data only (unseen levels -> global mean)
_y_all = (train[TARGET] == POSITIVE).astype(float)
_global_mean = float(_y_all.mean())
TE_SMOOTH = 25.0
_te_maps = {}
for _c in ["UniqueCarrier", "Origin", "Dest"]:
    _g = train.groupby(_c, dropna=False)[TARGET].apply(lambda s: (s == POSITIVE).sum())
    _n = train.groupby(_c, dropna=False)[TARGET].size()
    _te = (_g + TE_SMOOTH * _global_mean) / (_n + TE_SMOOTH)
    _te_maps[_c] = _te


def _route(df: pd.DataFrame) -> pd.Series:
    return df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


route_levels = pd.Index(sorted(_route(train).unique()))

# calendar helpers (2005/2006 are non-leap)
_CUM = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def _to_num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s.astype(str).str[2:], errors="coerce").to_numpy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # ALL feature engineering belongs here: predict_proba() calls prepare() on unseen rows, so anything you
    # compute on `train`/`evald` outside this function will NOT be applied to the hidden holdout.
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])  # unseen levels -> NaN
    # X["route"] = pd.Categorical(_route(df), categories=route_levels)  # ABLATED: hurt eval AUC
    mon = _to_num(df["Month"])
    dom = _to_num(df["DayofMonth"])
    dow = _to_num(df["DayOfWeek"])
    X["doy"] = _CUM[np.clip(mon, 1, 12).astype(int) - 1] + dom
    X["is_weekend"] = (dow >= 6).astype(int)
    dt = pd.to_numeric(df["DepTime"], errors="coerce").fillna(0).to_numpy()
    hod = (dt // 100) % 24
    X["redeye"] = ((hod >= 21) | (hod <= 4)).astype(int)
    for _c, _m in _te_maps.items():
        X[_c + "_te"] = df[_c].map(_m).astype(float).fillna(_global_mean)
    return X


def to_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POSITIVE).astype(int).to_numpy()


# --- model --------------------------------------------------------------------
params = dict(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=20,
    gamma=1.0,
    reg_lambda=5.0,
    subsample=0.9,
    colsample_bytree=0.9,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=SEED,
    n_jobs=N_JOBS,
)

t0 = time.time()
X = prepare(train)
y = to_y(train)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
es_model = xgb.XGBClassifier(**params)
es_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
best_n = int(es_model.best_iteration) + 1
print(f"ES training time: {time.time() - t0:.1f}s, best_iter={best_n}, es_auc={es_model.best_score:.4f}")

t0 = time.time()
model = xgb.XGBClassifier(**{**params, "n_estimators": best_n, "early_stopping_rounds": None})
model.fit(X, y)
print(f"Final training time: {time.time() - t0:.1f}s ({model.n_estimators} trees)")


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(df))[:, 1]


t0 = time.time()
eval_auc = roc_auc_score(to_y(evald), predict_proba(evald))
print(f"Eval time: {time.time() - t0:.1f}s")
print(f"Eval AUC: {eval_auc:.4f}")
